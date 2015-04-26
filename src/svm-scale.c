#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "svm-scale.h"

char *line = NULL;
int max_line_len = 1024;
double lower=-1.0,upper=1.0,y_lower,y_upper;
int y_scaling = 0;
double *feature_max;
double *feature_min;
double y_max = -DBL_MAX;
double y_min = DBL_MAX;
int max_index;
int min_index;
long int num_nonzeros = 0;
long int new_num_nonzeros = 0;

#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))

void output_target(double value, FILE *fp);
void output(int index, double value, FILE *fp);
char* readline(FILE *input);
int clean_up(FILE *fp_restore, FILE *fp, const char *msg);

int svm_scale(const char* data_file, const char* dst_file,
              const double _lower, const double _upper,
              const double _y_lower, const double _y_upper)
{
	int i,index;
	FILE *fp, *fp_restore = NULL;
    
    lower = _lower;
    upper = _upper;
	y_lower = _y_lower;
    y_upper = _y_upper;
    if (y_lower != 0 && y_upper != 0) y_scaling = 1;

	if(!(upper > lower) || (y_scaling && !(y_upper > y_lower)))
	{
		fprintf(stderr,"inconsistent lower/upper specification\n");
		return 1;
	}

	fp=fopen(data_file,"r");
	
	if(fp==NULL)
	{
		fprintf(stderr,"can't open file %s\n", data_file);
		return 1;
	}

	line = (char *) malloc(max_line_len*sizeof(char));

#define SKIP_TARGET\
	while(isspace(*p)) ++p;\
	while(!isspace(*p)) ++p;

#define SKIP_ELEMENT\
	while(*p!=':') ++p;\
	++p;\
	while(isspace(*p)) ++p;\
	while(*p && !isspace(*p)) ++p;
	
	/* assumption: min index of attributes is 1 */
	/* pass 1: find out max index of attributes */
	max_index = 0;
	min_index = 1;

	while(readline(fp)!=NULL)
	{
		char *p=line;

		SKIP_TARGET

		while(sscanf(p,"%d:%*f",&index)==1)
		{
			max_index = max(max_index, index);
			min_index = min(min_index, index);
			SKIP_ELEMENT
			num_nonzeros++;
		}
	}

	if(min_index < 1)
		fprintf(stderr,
			"WARNING: minimal feature index is %d, but indices should start from 1\n", min_index);

	rewind(fp);

	feature_max = (double *)malloc((max_index+1)* sizeof(double));
	feature_min = (double *)malloc((max_index+1)* sizeof(double));

	if(feature_max == NULL || feature_min == NULL)
	{
		fprintf(stderr,"can't allocate enough memory\n");
		return 1;
	}

	for(i=0;i<=max_index;i++)
	{
		feature_max[i]=-DBL_MAX;
		feature_min[i]=DBL_MAX;
	}

	/* pass 2: find out min/max value */
	while(readline(fp)!=NULL)
	{
		char *p=line;
		int next_index=1;
		double target;
		double value;

		if (sscanf(p,"%lf",&target) != 1)
			return clean_up(fp_restore, fp, "ERROR: failed to read labels\n");
		y_max = max(y_max,target);
		y_min = min(y_min,target);
		
		SKIP_TARGET

		while(sscanf(p,"%d:%lf",&index,&value)==2)
		{
			for(i=next_index;i<index;i++)
			{
				feature_max[i]=max(feature_max[i],0);
				feature_min[i]=min(feature_min[i],0);
			}
			
			feature_max[index]=max(feature_max[index],value);
			feature_min[index]=min(feature_min[index],value);

			SKIP_ELEMENT
			next_index=index+1;
		}		

		for(i=next_index;i<=max_index;i++)
		{
			feature_max[i]=max(feature_max[i],0);
			feature_min[i]=min(feature_min[i],0);
		}	
	}

	rewind(fp);

	/* pass 3: scale */
    FILE *fp_dist = fopen(dst_file, "w");
	while(readline(fp)!=NULL)
	{
		char *p=line;
		int next_index=1;
		double target;
		double value;
		
		if (sscanf(p,"%lf",&target) != 1)
			return clean_up(NULL, fp, "ERROR: failed to read labels\n");
		output_target(target, fp_dist);

		SKIP_TARGET

		while(sscanf(p,"%d:%lf",&index,&value)==2)
		{
			for(i=next_index;i<index;i++)
				output(i,0,fp_dist);
			
			output(index,value,fp_dist);

			SKIP_ELEMENT
			next_index=index+1;
		}		

		for(i=next_index;i<=max_index;i++)
			output(i,0,fp_dist);

        fprintf(fp_dist, "\n");
	}
    fclose(fp_dist);

	if (new_num_nonzeros > num_nonzeros)
		fprintf(stderr, 
			"WARNING: original #nonzeros %ld\n"
			"         new      #nonzeros %ld\n"
			"Use -l 0 if many original feature values are zeros\n",
			num_nonzeros, new_num_nonzeros);

	free(line);
	free(feature_max);
	free(feature_min);
	fclose(fp);
	return 0;
}

char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void output_target(double value, FILE *fp)
{
	if(y_scaling)
	{
		if(value == y_min)
			value = y_lower;
		else if(value == y_max)
			value = y_upper;
		else value = y_lower + (y_upper-y_lower) *
			     (value - y_min)/(y_max-y_min);
	}
    fprintf(fp, "%g ", value);
}

void output(int index, double value, FILE *fp)
{
	/* skip single-valued attribute */
	if(feature_max[index] == feature_min[index])
		return;

	if(value == feature_min[index])
		value = lower;
	else if(value == feature_max[index])
		value = upper;
	else
		value = lower + (upper-lower) * 
			(value-feature_min[index])/
			(feature_max[index]-feature_min[index]);

	if(value != 0)
	{
        fprintf(fp, "%d:%g ", index, value);
		new_num_nonzeros++;
	}
}

int clean_up(FILE *fp_restore, FILE *fp, const char* msg)
{
	fprintf(stderr,	"%s", msg);
	free(line);
	free(feature_max);
	free(feature_min);
	fclose(fp);
	if (fp_restore)
		fclose(fp_restore);
	return -1;
}

