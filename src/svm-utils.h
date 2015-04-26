#pragma once

#include "ofMain.h"
#include "svm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
static char *line = NULL;
static int max_line_len;

static void exit_input_error(int line_num)
{
    ofLogError("ofxSvm") << "Wrong input format at line " << line_num;
}


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


static int read_problem(const char *filename, struct svm_problem* prob, struct svm_parameter* param)
{
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;
    struct svm_node *x_space;
    
    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        fclose(fp);
        return 1;
    }
    
    prob->l = 0;
    elements = 0;
    
    max_line_len = 1024;
    line = Malloc(char, max_line_len);
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
        ++prob->l;
    }
    rewind(fp);
    
    prob->y = Malloc(double,prob->l);
    prob->x = Malloc(struct svm_node *,prob->l);
    x_space = Malloc(struct svm_node,elements);
    
    max_index = 0;
    j=0;
    for(i=0;i<prob->l;i++)
    {
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        readline(fp);
        prob->x[i] = &x_space[j];
        label = strtok(line," \t\n");
        if(label == NULL) // empty line
        {
            exit_input_error(i+1);
            goto ERROR_RETURN;
        }
        
        prob->y[i] = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0')
        {
            exit_input_error(i+1);
            goto ERROR_RETURN;
        }
        
        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");
            
            if(val == NULL)
                break;
            
            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
            {
                exit_input_error(i+1);
                goto ERROR_RETURN;
            }
            else inst_max_index = x_space[j].index;
            
            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
            {
                exit_input_error(i+1);
                goto ERROR_RETURN;
            }
            
            ++j;
        }
        
        if(inst_max_index > max_index)
            max_index = inst_max_index;
        x_space[j++].index = -1;
    }
    
    if(param->gamma == 0 && max_index > 0)
        param->gamma = 1.0/max_index;
    
    if(param->kernel_type == PRECOMPUTED)
        for(i=0;i<prob->l;i++)
        {
            if (prob->x[i][0].index != 0)
            {
                ofLogError("ofxSvm::read_problem") << "Wrong input format: first column must be 0:sample_serial_number";
                goto ERROR_RETURN;
            }
            if ((int)prob->x[i][0].value <= 0 || (int)prob->x[i][0].value > max_index)
            {
                ofLogError("ofxSvm::read_problem") << "Wrong input format: sample_serial_number out of range";
                goto ERROR_RETURN;
            }
        }
    
    free(x_space);
    fclose(fp);
    return 0;
    
ERROR_RETURN:
    free(x_space);
    fclose(fp);
    return 1;
}
