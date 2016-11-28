/* mex Function beninterp2_for_mat.c 
//OutputMat = beninterp2_for_mat(ypointsvec, vMat, InputMat, VofyMin, yMin, VofyMaxVec, yMax);

//Modified 26/July/2011 by Benjamin kay to take a matrix as an argument. Also made a tweak to improve performance for low values (strictly less than min to trigger extrap) 
// tweaked on 8/6/11
// Big overhaul on 8/11/11
 * This takes a (possibly three dimensional) array of input values InputMat and tries to interpolate them. It looks the input values up on the vector ypointsvec.
 * Depending on which page it is in, it looks up the associated value in vMat which must have the same number of rows as ypointsvec and the same number of columns 
 * as the pages of InputMat. The min extrapolation value is common, the max extrapolation is a vector of values with a length equal to the number of columns. 
 * General idea: look up a bunch of values. Everything on page 1 looked up against column 1 of vMat. Everything on page 2 looked up against column 2 of vMat. 
 * In my problem, the columns of InputMat are ytplus1 outcomes for various market returns. The pages of InputMat are the various outcomes of the housing process. 
 *
 *Simple full example: 
 * beninterp2_for_mat((0:5)',bsxfun(@times,(0:5)',[1,1.1,1.2]),bsxfun(@times,ones(3,3,3),[1 2 3]),-1,-2,[5.1,5.6,6.1]',5)

ans(:,:,1) =

     1     2     3
     1     2     3
     1     2     3


ans(:,:,2) =

    1.1000    2.2000    3.3000
    1.1000    2.2000    3.3000
    1.1000    2.2000    3.3000


ans(:,:,3) =

    1.2000    2.4000    3.6000
    1.2000    2.4000    3.6000
    1.2000    2.4000    3.6000


*/
#include "mex.h"
#include "matrix.h"
#include "math.h"


void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
    const mxArray *InputMat, *ypointsvec;
    mxArray *OutputMat;
    const mwSize *pDims;
    size_t ypointsCount, ColumnCount, RowCount, i, j, k, p, i1, i9, imid, PageCount, i1p, i9p;
    double *InputMatptr, *vMatptr, *ypointsvecptr, *OutputMatptr, *VofyMaxVecptr;
    double InputMatk;
    mwSize dims[2];
    /*/ added by me for extrap value*/
    double VofyMin, yMin, yMax;
    VofyMin  = mxGetScalar(prhs[3]);
    yMin  = mxGetScalar(prhs[4]);
    VofyMaxVecptr  = mxGetPr(prhs[5]);  
    yMax  = mxGetScalar(prhs[6]);
    
    /*/ Get inputs and dimensions*/
    ypointsvec = prhs[0];
    ypointsCount = mxGetM(ypointsvec); /* number of rows in matrix*/
    InputMat = prhs[2]; /* number(s) to find interpolated value f(InputMat) */
    /*RowCount = mxGetM(InputMat);*/ /* row numbers so you can interpolate a vector -- col length*/
    /*ColumnCount = mxGetN(InputMat);*/ /*Number of columns in each row of data -- row length*/
    pDims = mxGetDimensions(InputMat);
    
    RowCount = pDims[0];
    ColumnCount = pDims[1];
    if (mxGetNumberOfDimensions(InputMat) == 3)
    {
    /*printf("Number of pages %i\n",pDims[2]);   */     
    PageCount = pDims[2];
    dims[2] = PageCount;
    
    dims[0] = RowCount; dims[1] = ColumnCount;     
    plhs[0] = OutputMat = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);    
    }
    else        
    {
    PageCount = 1;
    dims[0] = RowCount; dims[1] = ColumnCount;         
    plhs[0] = OutputMat = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);    
    }
    /*/ Create output OutputMat */


    
    if (OutputMat==NULL) /*/ Cannot allocate memory*/
    {
        /*/ Return empty array*/
        dims[0] = 0; dims[1] = 0; dims[2] = 0;
        plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        return;
    }
    OutputMatptr = mxGetPr(OutputMat); /*This is the output data. */
    
    /*/ Get pointers*/
    InputMatptr = mxGetPr(InputMat);
    vMatptr = mxGetPr(prhs[1]);
    ypointsvecptr = mxGetPr(ypointsvec);
    /**/
    for(p=0;p < PageCount;p++) 
    {
        for(i=0;i < ColumnCount;i++)
        {
            for(j=0;j < RowCount;j++)
            {
                k = (p*RowCount*ColumnCount)+(i*RowCount)+j;
                /*outArray[(i*colLen)+j] = 2*xValues[(i*colLen)+j];*/
                /*/ Get data value*/
                InputMatk = InputMatptr[k];
                if (InputMatk < yMin)
                {
                    
                    OutputMatptr[k] = VofyMin;
                }
                else
                {
                    if (InputMatk > yMax)
                    {
                        OutputMatptr[k] = VofyMaxVecptr[p];
                    }
                    else
                    {
                        i1=0;
                        i9=ypointsCount-1; 
                        while (i9>i1+1) /*/ Dichotomy search*/
                        {
                            imid = (i1+i9+1)/2;
                            if (ypointsvecptr[imid]<InputMatk) i1=imid;
                            else i9=imid;
                        } /*/ of while loop*/
                        i1p = i1 + p * ypointsCount;
                        i9p = i9 + p * ypointsCount;
                        if (i1==i9)
                            OutputMatptr[k] = vMatptr[i1p];
                        else
                            OutputMatptr[k] = vMatptr[i1p] + (vMatptr[i9p]-vMatptr[i1p])*(InputMatk-ypointsvecptr[i1])/(ypointsvecptr[i9]-ypointsvecptr[i1]);
                    }
                }            
            }
        }     
 
    }
    return;
        
}

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