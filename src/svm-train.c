#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include "svm-train.h"
#include "svm-utils.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
int read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

int execute_svm_train(const char* _data_file_name, const char* _model_file_name)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
    int res;
    strcpy(input_file_name, _data_file_name);
    strcpy(model_file_name, _model_file_name);

	res = read_problem(input_file_name);
    if (res != 0)
        return 1;
	error_msg = svm_check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		return 1;
	}

	if(cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		model = svm_train(&prob,&param);
		if(svm_save_model(model_file_name,model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			return 1;
		}
		svm_free_and_destroy_model(&model);
	}
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}

void set_svm_train_parameters(const int _svm_type,
                              const int _kernel_type,
                              const int _degree,
                              const double _gamma,
                              const double _coef0,
                              const double _nu,
                              const double _cache_size,
                              const double _C,
                              const double _eps,
                              const double _p,
                              const int _shrinking,
                              const int _probability,
                              const int _cross_validation_nr)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

    param.svm_type = _svm_type;
    param.kernel_type = _kernel_type;
	param.degree = _degree;
    param.gamma = _gamma;
	param.coef0 = _coef0;
	param.nu = _nu;
	param.cache_size = _cache_size;
	param.C = _C;
	param.eps = _eps;
	param.p = _p;
	param.shrinking = _shrinking;
    param.probability = _probability;
    if (_cross_validation_nr != 0)
    {
        cross_validation = 1;
        nr_fold = _cross_validation_nr;
        if(nr_fold < 2)
        {
            fprintf(stderr,"n-fold cross validation: n must >= 2\n");
            return 1;
        }
    }
	svm_set_print_string_function(print_func);
}

void set_svm_train_weight(const int lavel, const double weight)
{
    ++param.nr_weight;
	param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
	param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
	param.weight_label[param.nr_weight-1] = lavel;
	param.weight[param.nr_weight-1] = weight;
}

// read in a problem (in svmlight format)

int read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		return 1;
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			return exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			return exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				return exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				return exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				return 1;
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				return 1;
			}
		}

	fclose(fp);
    return 0;
}
