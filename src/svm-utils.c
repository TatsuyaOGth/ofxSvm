#include <stdio.h>
#include "svm-utils.h"

int exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    return 1;
}
