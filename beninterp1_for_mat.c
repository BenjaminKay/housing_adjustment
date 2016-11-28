/* mex Function beninterp1_for_mat.c 
//////////////////////////////////////////////////////////////////////////
//mex nakeinterp1.c
// mex function nakeinterp1.c
// Dichotomy search of indices
// Calling:
// idx=nakeinterp1(x, y, xi);
// where x, y and xi are double column vectors
// x must be sorted in ascending order; x and y have the same length
// NO ARGUMENT CHECKING
// Compile:
// mex -O -v nakeinterp1.c
// Author: Bruno Luong
// Original: 19/Feb/2009
//Modified on 04/Mar/2011 by Benjamin Kay to take a minimum and maximum value to use extrapvalue on
//beninterp1(x, y, xi,f(xmin),xmin,f(xmax),xmax);

//Modified 26/July/2011 by Benjamin kay to take a matrix as an argument. Also made a tweak to improve performance for low values (strictly less than min to trigger extrap) 
// tweaked on 8/6/11
//////////////////////////////////////////////////////////////////////////
*/
#include "mex.h"
#include "matrix.h"
#include "math.h"

/* // Gateway routine*/
/* The gateway header stuff is int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[], we will's document but to summarize
 Gateway can include four things:
– prhs, nrhs, plhs, nlhs
– RHS – input argument, from matlab to C
– LHS – output arguments, from C to matlab
– P – pointer to where argument lives (must
include)
– N – number of arguments
*/
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
    const mxArray *xi, *xgrid;
    mxArray *idx;
    size_t nx, n, m, i, j, k, i1, i9, imid;
    double *xiptr, *yptr, *xgridptr, *idxptr;
    double xik;
    mwSize dims[2];
    /*/ added by me for extrap value*/
    double extrapvalmin,extrapvalmax, extrapmin, extrapmax;
    extrapvalmin  = mxGetScalar(prhs[3]);
    extrapmin  = mxGetScalar(prhs[4]);
    extrapvalmax  = mxGetScalar(prhs[5]);    
    extrapmax  = mxGetScalar(prhs[6]);
    
    /*/ Get inputs and dimensions*/
    xgrid = prhs[0];
    nx = mxGetM(xgrid); /* number of rows in matrix*/
    xi = prhs[2]; /* number(s) to find interpolated value f(xi) */
    m = mxGetM(xi); /* row numbers so you can interpolate a vector -- col length*/
    n = mxGetN(xi); /*Number of columns in each row of data -- row length*/
    
    /*/ Create output idx */
    dims[0] = m; dims[1] = n;
    plhs[0] = idx = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    if (idx==NULL) /*/ Cannot allocate memory*/
    {
        /*/ Return empty array*/
        dims[0] = 0; dims[1] = 0;
        plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
        return;
    }
    idxptr = mxGetPr(idx); /*This is the output data. */
    
    /*/ Get pointers*/
    xiptr = mxGetPr(xi);
    yptr = mxGetPr(prhs[1]);
    xgridptr = mxGetPr(xgrid);
 
    for(i=0;i < n;i++)
    {
        for(j=0;j < m;j++)
        {
            k = (i*m)+j;
            /*outArray[(i*colLen)+j] = 2*xValues[(i*colLen)+j];*/
            /*/ Get data value*/
            xik = xiptr[k];
            if (xik < extrapmin)
            {
                idxptr[k] = extrapvalmin;
            }
            else
            {
                if (xik > extrapmax)
                {
                    idxptr[k] = extrapvalmax;
                }
                else
                {
                    i1=0;
                    i9=nx-1; 
                    while (i9>i1+1) /*/ Dichotomy search*/
                    {
                        imid = (i1+i9+1)/2;
                        if (xgridptr[imid]<xik) i1=imid;
                        else i9=imid;
                    } /*/ of while loop*/
                    if (i1==i9)
                        idxptr[k] = yptr[i1];
                    else
                        idxptr[k] = yptr[i1] + (yptr[i9]-yptr[i1])*(xik-xgridptr[i1])/(xgridptr[i9]-xgridptr[i1]);
                }
            }            
        }
    }    
    return;
        
}