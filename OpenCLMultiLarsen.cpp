#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <cstring>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/opencl.h>
#define MAX_SOURCE_SIZE (0x100000)  
#define INF 50000
#define N 1500

//N is the number of models to solve in parallel.

using namespace std;
size_t localSize;
int X_elements;
int Y_elements; 
    // Number of total work items - localSize must be devisor
  size_t globalSize ;
//siduals and step as txt file.
int write(cl_mem d_beta, cl_mem d_step, cl_mem d_upper1, int M, int Z, int st, int l, cl_command_queue queue){
int	i, j, act;
double	*h_beta = new double[(Z - 1) * l];
double	*h_upper1 = new double[l];
int	*h_step = new int[l];

ofstream f, o, r;
	
	if(st != 0)f.open("OPCBeta.csv", ios::out | ios::app);
	else f.open("OPCBeta.csv", ios::out);
	if(st != 0)o.open("OPCStep.txt", ios::out | ios::app);
        else o.open("OPCStep.txt", ios::out);
	if(st != 0)r.open("OPCRess.txt", ios::out | ios::app);
        else r.open("OPCRess.txt", ios::out);
	//cudaMemcpy(h_beta, d_beta, (Z - 1) * l * sizeof(double), cudaMemcpyDeviceToHost);
        clEnqueueReadBuffer(queue, d_beta, CL_TRUE, 0,(Z - 1) * l * sizeof(double), h_beta, 0, NULL, NULL );
	//cudaMemcpy(h_step, d_step, l * sizeof(int), cudaMemcpyDeviceToHost);
        clEnqueueReadBuffer(queue, d_step, CL_TRUE, 0,l * sizeof(int), h_step, 0, NULL, NULL );
	//cudaMemcpy(h_upper1, d_upper1, l * sizeof(double), cudaMemcpyDeviceToHost);
        clEnqueueReadBuffer(queue, d_upper1, CL_TRUE, 0, l * sizeof(double), h_upper1, 0, NULL, NULL );
       
	for(i = 0;i < l;i++){
		o << h_step[i] - 1 << '\n';
		r << 1 - h_upper1[i] << '\n';
		act = st + i;
		for(j = 0;j < Z - 1;j++){
			if(fabs(h_beta[j * l + i]) != 0){
				if(j < act){
					f << act << ',' << j << ',' << h_beta[j * l + i] << '\n';
				}
				else{
					f << act << ',' << j + 1 << ',' << h_beta[j * l + i] << '\n';
				}
			}
		}
	}
	delete[] h_beta;
	delete[] h_upper1;
	delete[] h_step;
	f.close();
	o.close();
	r.close();
	return 0;
}
/*
X input(M - 1, Z)
Y output(M - 1, Z)
st is the starting model and l is the number of models from starting to solve.
*/
int lars(cl_mem d_X, cl_mem d_Y, cl_mem d_mu, cl_mem d_c, cl_mem d_, cl_mem d_G, cl_mem d_I,
		 cl_mem d_beta, cl_mem d_betaOLS, cl_mem d_d, cl_mem d_gamma, cl_mem d_cmax, 
		 cl_mem d_upper1, cl_mem d_normb, cl_mem d_lVars, cl_mem d_nVars, cl_mem d_ind,
		  cl_mem d_step, cl_mem d_done, cl_mem d_lasso, cl_mem d_ctrl, 
		  int M, int Z, int st, int l,
		  cl_int err, cl_kernel kernel[], cl_command_queue queue, cl_mem d_lambda, cl_mem d_val){
int	n = min(M - 1, Z - 1), i, j;
cout<<"n: "<<n<<endl;
int step = 1;
int cont = 1;


size_t bl = 1000;
size_t gl = ((l+1000-1)/1000)*1000;

size_t bZl[2] = {31,31};
size_t gZl[2] = {((Z+31-1)/31)*31,((l+31-1)/31)*31};

size_t bMl[2] = {31,31};
size_t gMl[2] = {((M+31-1)/31)*31, ((l+31-1)/31)*31};

size_t bnl[2] = {31,31};
size_t gnl[2] = {((n+31-1)/31)*31, ((l+31-1)/31)*31};




int	*h_ctrl = new int, *h_nVars = new int[l], *h_done = new int[l];
int *h_step = new int[l];
	double *G = new double[l];
	
	
		double *set_0 = new double[(M-1)*l];
		memset(set_0,0,sizeof(double)*(M-1)*l);
		double val = 0;
		err = clEnqueueWriteBuffer(queue, d_mu, CL_TRUE, 0,
                                   sizeof(double)*(M-1)*l, set_0, 0, NULL, NULL);
		delete[] set_0;
		set_0 = new double[(Z-1)*l];
		memset(set_0,0,sizeof(double)*(Z-1)*l);
		 err = clEnqueueWriteBuffer(queue, d_beta, CL_TRUE, 0,
                                   sizeof(double)*(Z-1)*l, set_0, 0, NULL, NULL);

		delete[] set_0;
		set_0 = new double[l];
		memset(set_0,0,sizeof(double)*l);
		 
		 err = clEnqueueWriteBuffer(queue, d_lambda, CL_TRUE, 0,
                                   sizeof(double)*l, set_0, 0, NULL, NULL);

		delete[] set_0;
    	
    
   
   
	//dInit<<<gl, bl>>>(d_nVars, d_lasso, d_done, d_step, l);	
		 	
	
		
int *set_val = new int;

                   
    err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &d_nVars);
    err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &d_lasso);
    err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), &d_done);
    err |= clSetKernelArg(kernel[2], 3, sizeof(cl_mem), &d_step);
    err |= clSetKernelArg(kernel[2], 4, sizeof( int), &l);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[2], 1, NULL,  &gl, &bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
   
memset(h_step,0,sizeof(int)*l);
clEnqueueReadBuffer(queue, d_step, CL_TRUE, 0, sizeof(int)*l, h_step, 0, NULL, NULL );
cout<<"init: "<<h_step[45]<<"  "<<h_step[46]<<"  "<<h_step[47]<<endl;
step = 1;
 double lambda = 0.43;
	while(step < 8*n){
		cout<<"Step "<< step<<endl;
		
		memset(set_val,0,sizeof(int));
		err = clEnqueueWriteBuffer(queue, d_ctrl, CL_TRUE, 0,
                                   sizeof(int), set_val, 0, NULL, NULL);
	//	dCheck<<<gl, bl>>>(d_ctrl, d_step, d_nVars, n, l, d_done);
		//check kernel
	err  = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &d_ctrl);
    err |= clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &d_step);
    err |= clSetKernelArg(kernel[3], 2, sizeof(cl_mem), &d_nVars);
    err |= clSetKernelArg(kernel[3], 3, sizeof( int), &n);
    err |= clSetKernelArg(kernel[3], 4, sizeof( int), &l);
    err |= clSetKernelArg(kernel[3], 5, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL,  &gl, &bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
    
   
	
	//cout<<"hello"<<endl;
		//cudaMemcpy(h_ctrl, d_ctrl, sizeof(int), cudaMemcpyDeviceToHost);
 clEnqueueReadBuffer(queue, d_ctrl, CL_TRUE, 0, sizeof(int), h_ctrl, 0, NULL, NULL );
 //cout<<"ctrl: "<<h_ctrl[0]<<endl;
		if(h_ctrl[0] == 0){
		//	cout<<"ctrl break"<<endl;
			break;
		}


		//dCorr<<<gZl, bZl>>>(d_X, d_Y, d_mu, d_c, d_, M, Z, st, l, d_done);
		err  = clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &d_X);
    	err |= clSetKernelArg(kernel[4], 1, sizeof(cl_mem), &d_Y);
   	 	err |= clSetKernelArg(kernel[4], 2, sizeof(cl_mem), &d_mu);
   	 	err |= clSetKernelArg(kernel[4], 3, sizeof(cl_mem), &d_c);
   	 	err |= clSetKernelArg(kernel[4], 4, sizeof(cl_mem), &d_);
    	err |= clSetKernelArg(kernel[4], 5, sizeof( int), &M);
    	err |= clSetKernelArg(kernel[4], 6, sizeof( int), &Z);
    	err |= clSetKernelArg(kernel[4], 7, sizeof( int), &st);
    	err |= clSetKernelArg(kernel[4], 8, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[4], 9, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[4], 2, NULL,  gZl, bZl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);



	//cout<<"hello"<<endl;
		//dExcCorr<<<gnl, bnl>>>(d_, d_lVars, d_nVars, l, d_done);
		err  = clSetKernelArg(kernel[5], 0, sizeof(cl_mem), &d_);
    	err |= clSetKernelArg(kernel[5], 1, sizeof(cl_mem), &d_lVars);
   	 	err |= clSetKernelArg(kernel[5], 2, sizeof(cl_mem), &d_nVars);
   	  	err |= clSetKernelArg(kernel[5], 3, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[5], 4, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[5], 2, NULL,  gnl, bnl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    

    


	//cout<<"hello"<<endl;
		//dMaxcorr<<<gl, bl>>>(d_, d_cmax, d_ind, Z, l, d_done);
   		err  = clSetKernelArg(kernel[6], 0, sizeof(cl_mem), &d_);
		err |= clSetKernelArg(kernel[6], 1, sizeof(cl_mem), &d_cmax);
    	err |= clSetKernelArg(kernel[6], 2, sizeof(cl_mem), &d_ind);
     	err |= clSetKernelArg(kernel[6], 3, sizeof( int), &Z);
       	err |= clSetKernelArg(kernel[6], 4, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[6], 5, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[6], 1, NULL,  &gl, &bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

   

		//dLassoAdd<<<gl, bl>>>(d_ind, d_lVars, d_nVars, d_lasso, l, d_done);
 	   	err  = clSetKernelArg(kernel[7], 0, sizeof(cl_mem), &d_ind);
    	err |= clSetKernelArg(kernel[7], 1, sizeof(cl_mem), &d_lVars);
   	 	err |= clSetKernelArg(kernel[7], 2, sizeof(cl_mem), &d_nVars);
   	 	err |= clSetKernelArg(kernel[7], 3, sizeof(cl_mem), &d_lasso);
    	err |= clSetKernelArg(kernel[7], 4, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[7], 5, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[7], 1, NULL,  &gl, &bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);


    
	//cout<<"hello"<<endl;
		//dXincTY<<<gnl, bnl>>>(d_X, d_Y, d_, d_lVars, d_nVars, M, Z, st, l, d_done);
    	err  = clSetKernelArg(kernel[8], 0, sizeof(cl_mem), &d_X);
    	err |= clSetKernelArg(kernel[8], 1, sizeof(cl_mem), &d_Y);
   	 	err |= clSetKernelArg(kernel[8], 2, sizeof(cl_mem), &d_);
   	 	err |= clSetKernelArg(kernel[8], 3, sizeof(cl_mem), &d_lVars);
    	err |= clSetKernelArg(kernel[8], 4, sizeof(cl_mem), &d_nVars);
    	err |= clSetKernelArg(kernel[8], 5, sizeof( int), &M);
    	err |= clSetKernelArg(kernel[8], 6, sizeof( int), &Z);
    	err |= clSetKernelArg(kernel[8], 7, sizeof( int), &st);
    	err |= clSetKernelArg(kernel[8], 8, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[8], 9, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[8], 2, NULL,  gnl, bnl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    
	//cout<<"hello"<<endl;
	//cudaMemcpy(h_nVars, d_nVars, l * sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_done, d_done, l * sizeof(int), cudaMemcpyDeviceToHost);	
    		clEnqueueReadBuffer(queue, d_nVars, CL_TRUE, 0, sizeof(int)*l, h_nVars, 0, NULL, NULL );
		clEnqueueReadBuffer(queue, d_done, CL_TRUE, 0, sizeof(int)*l, h_done, 0, NULL, NULL );
		for(j = 0;j < l;j++)
		{
		
			if(h_done[j]){
			//	cout<<"continue done cont = "<<cont<<endl;
				cont++;
				continue;
			}
			//dSetGram<<<h_nVars[j], h_nVars[j]>>>(d_X, d_G, d_I, d_lVars, h_nVars[j], M, Z, j, st, l);

			
    size_t h_bl = h_nVars[j];
    size_t h_gl = (h_nVars[j])*h_nVars[j];
    size_t unit = 1;

		err  = clSetKernelArg(kernel[9], 0, sizeof(cl_mem), &d_X);
    	err |= clSetKernelArg(kernel[9], 1, sizeof(cl_mem), &d_G);
   	 	err |= clSetKernelArg(kernel[9], 2, sizeof(cl_mem), &d_I);
   	 	err |= clSetKernelArg(kernel[9], 3, sizeof(cl_mem), &d_lVars);
    	err |= clSetKernelArg(kernel[9], 4, sizeof( int), &h_nVars[j]);
    	err |= clSetKernelArg(kernel[9], 5, sizeof( int), &M);
    	err |= clSetKernelArg(kernel[9], 6, sizeof( int), &Z);
    	err |= clSetKernelArg(kernel[9], 7, sizeof( int), &j);
    	err |= clSetKernelArg(kernel[9], 8, sizeof( int), &st);
    	err |= clSetKernelArg(kernel[9], 9, sizeof( int), &l);
    
    	 
     // Execute the kernel over the entire range of the data set
    
      	//cout<<"hello"<<endl;

    err = clEnqueueNDRangeKernel(queue, kernel[9], 1, NULL,  &h_gl, &h_bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

   


		
	



			for(i = 0;i < h_nVars[j];i++)
			{
				  
				
				//nodiag_normalize<<<1, h_nVars[j]>>>(d_G, d_I, h_nVars[j], i);
		err  = clSetKernelArg(kernel[10], 0, sizeof(cl_mem), &d_G);
    	err |= clSetKernelArg(kernel[10], 1, sizeof(cl_mem), &d_I);	 	
   	 	err |= clSetKernelArg(kernel[10], 2, sizeof( int), &h_nVars[j]);
    	err |= clSetKernelArg(kernel[10], 3, sizeof( int), &i);
    	
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[10], 1, NULL,  &h_bl, &h_bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);




				//diag_normalize<<<1, 1>>>(d_G, d_I, h_nVars[j], i);
		err  = clSetKernelArg(kernel[11], 0, sizeof(cl_mem), &d_G);
    	err |= clSetKernelArg(kernel[11], 1, sizeof(cl_mem), &d_I);
    	err |= clSetKernelArg(kernel[11], 2, sizeof( int), &h_nVars[j]);
    	err |= clSetKernelArg(kernel[11], 3, sizeof( int), &i);
    	
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[11], 1, NULL, &unit, &unit,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);


	//cout<<"hello"<<endl;


 	
			//	gaussjordan<<<h_nVars[j], h_nVars[j]>>>(d_G, d_I, h_nVars[j], i);
		err  = clSetKernelArg(kernel[12], 0, sizeof(cl_mem), &d_G);
    	err |= clSetKernelArg(kernel[12], 1, sizeof(cl_mem), &d_I);
    	err |= clSetKernelArg(kernel[12], 2, sizeof( int), &h_nVars[j]);
    	err |= clSetKernelArg(kernel[12], 3, sizeof( int), &i);
    	
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[12], 1, NULL,  &h_gl, &h_bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

		//	set_zero<<<1, h_nVars[j]>>>(d_G, d_I, h_nVars[j], i);
		err  = clSetKernelArg(kernel[13], 0, sizeof(cl_mem), &d_G);
    	err |= clSetKernelArg(kernel[13], 1, sizeof(cl_mem), &d_I);
       	err |= clSetKernelArg(kernel[13], 2, sizeof( int), &h_nVars[j]);
    	err |= clSetKernelArg(kernel[13], 3, sizeof( int), &i);
    	
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[13], 1, NULL, &h_bl, &h_bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

}
				

	
		//	dBetaols<<<1, h_nVars[j]>>>(d_I, d_, d_betaOLS, h_nVars[j], j, l);	
		err  = clSetKernelArg(kernel[14], 0, sizeof(cl_mem), &d_I);
    	err |= clSetKernelArg(kernel[14], 1, sizeof(cl_mem), &d_);
   	  	err |= clSetKernelArg(kernel[14], 2, sizeof(cl_mem), &d_betaOLS);
   	  	err |= clSetKernelArg(kernel[14], 3, sizeof( int), &h_nVars[j]);
    	err |= clSetKernelArg(kernel[14], 4, sizeof( int), &j);
    	err |= clSetKernelArg(kernel[14], 5, sizeof( int), &l);
    	
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[14], 1, NULL,  &h_bl, &h_bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);


   

		}
	

   
	
		
		//ddgamma<<<gMl, bMl>>>(d_X, d_mu, d_beta, d_betaOLS, d_gamma, 
		//			d_d, d_lVars, d_nVars, M, Z, st, l, d_done);
		err  = clSetKernelArg(kernel[15], 0, sizeof(cl_mem), &d_X);
    	err |= clSetKernelArg(kernel[15], 1, sizeof(cl_mem), &d_mu);
   	 	err |= clSetKernelArg(kernel[15], 2, sizeof(cl_mem), &d_beta);
   	 	err |= clSetKernelArg(kernel[15], 3, sizeof(cl_mem), &d_betaOLS);
    	err |= clSetKernelArg(kernel[15], 4, sizeof(cl_mem), &d_gamma);
    	err |= clSetKernelArg(kernel[15], 5, sizeof(cl_mem), &d_d);
    	err |= clSetKernelArg(kernel[15], 6, sizeof(cl_mem), &d_lVars);
    	err |= clSetKernelArg(kernel[15], 7, sizeof(cl_mem), &d_nVars);
    	err |= clSetKernelArg(kernel[15], 8, sizeof( int), &M);
    	err |= clSetKernelArg(kernel[15], 9, sizeof( int), &Z);
    	err |= clSetKernelArg(kernel[15], 10, sizeof( int), &st);
    	err |= clSetKernelArg(kernel[15], 11, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[15], 12, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[15], 2, NULL, gMl, bMl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
    
    	
		//dGammamin<<<gl, bl>>>(d_gamma, d_ind, d_nVars, l, d_done);
		err  = clSetKernelArg(kernel[16], 0, sizeof(cl_mem), &d_gamma);
    	err |= clSetKernelArg(kernel[16], 1, sizeof(cl_mem), &d_ind);
   	 	err |= clSetKernelArg(kernel[16], 2, sizeof(cl_mem), &d_nVars);
   	  	err |= clSetKernelArg(kernel[16], 3, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[16], 4, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[16], 1, NULL, &gl, &bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);


  
		//dXTd<<<gZl, bZl>>>(d_X, d_c, d_, d_d, d_cmax, M, Z, st, l, d_done);
		err  = clSetKernelArg(kernel[17], 0, sizeof(cl_mem), &d_X);
    	err |= clSetKernelArg(kernel[17], 1, sizeof(cl_mem), &d_c);
   	 	err |= clSetKernelArg(kernel[17], 2, sizeof(cl_mem), &d_);
   	 	err |= clSetKernelArg(kernel[17], 3, sizeof(cl_mem), &d_d);
    	err |= clSetKernelArg(kernel[17], 4, sizeof(cl_mem), &d_cmax);
    	err |= clSetKernelArg(kernel[17], 5, sizeof( int), &M);
    	err |= clSetKernelArg(kernel[17], 6, sizeof( int), &Z);
    	err |= clSetKernelArg(kernel[17], 7, sizeof( int), &st);
    	err |= clSetKernelArg(kernel[17], 8, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[17], 9, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[17], 2, NULL, gZl, bZl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
//
		//dExctmp<<<gnl, bnl>>>(d_, d_lVars, d_nVars, l, d_done);
		err  = clSetKernelArg(kernel[18], 0, sizeof(cl_mem), &d_);
    	err |= clSetKernelArg(kernel[18], 1, sizeof(cl_mem), &d_lVars);
   	 	err |= clSetKernelArg(kernel[18], 2, sizeof(cl_mem), &d_nVars);
   	  	err |= clSetKernelArg(kernel[18], 3, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[18], 4, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[18], 2, NULL, gnl, bnl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

		//dTmpmin<<<gl, bl>>>(d_, Z, l, d_done);
		err  = clSetKernelArg(kernel[19], 0, sizeof(cl_mem), &d_);
      	err |= clSetKernelArg(kernel[19], 1, sizeof( int), &Z);
    	err |= clSetKernelArg(kernel[19], 2, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[19], 3, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[19], 1, NULL, &gl, &bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

		//dLassodev<<<gl, bl>>>(d_, d_gamma, d_nVars, d_lasso, n, l, d_done);
		err  = clSetKernelArg(kernel[20], 0, sizeof(cl_mem), &d_);
    	err |= clSetKernelArg(kernel[20], 1, sizeof(cl_mem), &d_gamma);
   	 	err |= clSetKernelArg(kernel[20], 2, sizeof(cl_mem), &d_nVars);
   	 	err |= clSetKernelArg(kernel[20], 3, sizeof(cl_mem), &d_lasso);
       	err |= clSetKernelArg(kernel[20], 4, sizeof( int), &n);
    	err |= clSetKernelArg(kernel[20], 5, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[20], 6, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[20], 1, NULL, &gl, &bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

		//dUpdate<<<gMl, bMl>>>(d_gamma, d_mu, d_beta, d_betaOLS, 
		//	d_d, d_lVars, d_nVars, M, l, d_done);
		err  = clSetKernelArg(kernel[21], 0, sizeof(cl_mem), &d_gamma);
    	err |= clSetKernelArg(kernel[21], 1, sizeof(cl_mem), &d_mu);
   	 	err |= clSetKernelArg(kernel[21], 2, sizeof(cl_mem), &d_beta);
   	 	err |= clSetKernelArg(kernel[21], 3, sizeof(cl_mem), &d_betaOLS);
    	err |= clSetKernelArg(kernel[21], 4, sizeof(cl_mem), &d_d);
    	err |= clSetKernelArg(kernel[21], 5, sizeof(cl_mem), &d_lVars);
       	err |= clSetKernelArg(kernel[21], 6, sizeof(cl_mem), &d_nVars);
    	err |= clSetKernelArg(kernel[21], 7, sizeof( int), &M);
       	err |= clSetKernelArg(kernel[21], 8, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[21], 9, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[21], 2, NULL, gMl, bMl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);	

 
   
    	size_t b_l = n;
    	size_t g_l = (l)*n;
		//dLassodrop<<<l, n>>>(d_ind, d_lVars, d_nVars, d_lasso, l, d_done);
		err  = clSetKernelArg(kernel[22], 0, sizeof(cl_mem), &d_ind);
    	err |= clSetKernelArg(kernel[22], 1, sizeof(cl_mem), &d_lVars);
   	 	err |= clSetKernelArg(kernel[22], 2, sizeof(cl_mem), &d_nVars);
   	 	err |= clSetKernelArg(kernel[22], 3, sizeof(cl_mem), &d_lasso);
       	err |= clSetKernelArg(kernel[22], 4, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[22], 5, sizeof(cl_mem), &d_done);
    	err |= clSetKernelArg(kernel[22], 6, sizeof(cl_mem), &d_val);
     // Execute the kernel over the entire range of the data set  

    
    err = clEnqueueNDRangeKernel(queue, kernel[22], 1, NULL, &g_l, &b_l,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);





		//dRess<<<gMl, bMl>>>(d_X, d_Y, d_, d_beta, M, Z, st, l, d_done);
		err  = clSetKernelArg(kernel[23], 0, sizeof(cl_mem), &d_X);
    	err |= clSetKernelArg(kernel[23], 1, sizeof(cl_mem), &d_Y);
   	 	err |= clSetKernelArg(kernel[23], 2, sizeof(cl_mem), &d_);
   	 	err |= clSetKernelArg(kernel[23], 3, sizeof(cl_mem), &d_beta);
      	err |= clSetKernelArg(kernel[23], 4, sizeof( int), &M);
    	err |= clSetKernelArg(kernel[23], 5, sizeof( int), &Z);
    	err |= clSetKernelArg(kernel[23], 6, sizeof( int), &st);
    	err |= clSetKernelArg(kernel[23], 7, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[23], 8, sizeof(cl_mem), &d_done);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[23], 2, NULL, gMl, bMl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

		//dFinal<<<gl, bl>>>(d_, d_beta, d_upper1, d_normb, d_nVars, d_step, 
		//	0.43, M, Z, l, d_done);
		//	cout<<"helloB"<<endl;
  
		err  = clSetKernelArg(kernel[24], 0, sizeof(cl_mem), &d_);
    	err |= clSetKernelArg(kernel[24], 1, sizeof(cl_mem), &d_beta);
   	 	err |= clSetKernelArg(kernel[24], 2, sizeof(cl_mem), &d_upper1);
   	 	err |= clSetKernelArg(kernel[24], 3, sizeof(cl_mem), &d_normb);
    	err |= clSetKernelArg(kernel[24], 4, sizeof(cl_mem), &d_nVars);
    	err |= clSetKernelArg(kernel[24], 5, sizeof(cl_mem), &d_step);
    	err |= clSetKernelArg(kernel[24], 6, sizeof(double), &lambda);
       	err |= clSetKernelArg(kernel[24], 7, sizeof( int), &M);
    	err |= clSetKernelArg(kernel[24], 8, sizeof( int), &Z);
       	err |= clSetKernelArg(kernel[24], 9, sizeof( int), &l);
    	err |= clSetKernelArg(kernel[24], 10, sizeof(cl_mem), &d_done);
    	err |= clSetKernelArg(kernel[24], 11, sizeof(cl_mem), &d_lambda);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[24], 1, NULL, &gl, &bl,0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    cout<<"done: "<<h_done[45]<<"  "<<h_done[46]<<" "<<h_done[47]<<endl;

    memset(G,0,sizeof(double)*l);
	clEnqueueReadBuffer(queue, d_lambda, CL_TRUE, 0, sizeof(double)*l, G, 0, NULL, NULL );
	cout<<"norm: "<<G[45]<<"  "<<G[46]<<"  "<<G[47]<<endl;



    memset(h_step,0,sizeof(int)*l);
clEnqueueReadBuffer(queue, d_step, CL_TRUE, 0, sizeof(int)*l, h_step, 0, NULL, NULL );
cout<<"final: "<<h_step[45]<<"  "<<h_step[46]<<"  "<<h_step[47]<<endl;
   
   
	//cout<<"helloA"<<endl;
		step++;
		
	
}
	
	delete[] h_ctrl;
	delete[] h_done;
	delete[] h_nVars;
	write(d_beta, d_step, d_upper1, M, Z, st, l,queue);
	cout << "Models " << st << "-->" << st + l << " completed... " << endl;

	return 0;
}


int main(int argc, char *argv[])
{

	FILE *fp1;
    const char fileName[] = "./kernel2.cl";
    size_t source_size;
    char *source_str;
        
    /* Load kernel source code */
    fp1 = fopen(fileName, "rb");
    if (!fp1) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
}
    
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp1);
    fclose(fp1);
   
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel[25];                 // kernel 
    cl_int err;
 
   
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the dfevice
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) &source_str, (const size_t *)&source_size, &err);
 
    // Build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel[0] = clCreateKernel(program, "dTrim", &err);
    kernel[1] = clCreateKernel(program, "dProc", &err);
    kernel[2] = clCreateKernel(program, "dInit", &err);
    kernel[3] = clCreateKernel(program, "dCheck", &err);
    kernel[4] = clCreateKernel(program, "dCorr", &err);
    kernel[5] = clCreateKernel(program, "dExcCorr", &err);
    kernel[6] = clCreateKernel(program, "dMaxcorr", &err);
    kernel[7] = clCreateKernel(program, "dLassoAdd", &err);
    kernel[8] = clCreateKernel(program, "dXincTY", &err);
    kernel[9] = clCreateKernel(program, "dSetGram", &err);
    kernel[10] = clCreateKernel(program, "nodiag_normalize", &err);
    kernel[11] = clCreateKernel(program, "diag_normalize", &err);
    kernel[12] = clCreateKernel(program, "gaussjordan", &err);
    kernel[13] = clCreateKernel(program, "set_zero", &err);
    kernel[14] = clCreateKernel(program, "dBetaols", &err);
    kernel[15] = clCreateKernel(program, "ddgamma", &err);
    kernel[16] = clCreateKernel(program, "dGammamin", &err);
    kernel[17] = clCreateKernel(program, "dXTd", &err);
    kernel[18] = clCreateKernel(program, "dExctmp", &err);
    kernel[19] = clCreateKernel(program, "dTmpmin", &err);
    kernel[20] = clCreateKernel(program, "dLassodev", &err);
    kernel[21] = clCreateKernel(program, "dUpdate", &err);
    kernel[22] = clCreateKernel(program, "dLassodrop", &err);
    kernel[23] = clCreateKernel(program, "dRess", &err);
    kernel[24] = clCreateKernel(program, "dFinal", &err);
   
    //kernel[25] = clCreateKernel(program, "set_mem", &err);

    fstream fp(argv[argc - 1]);
	int     M, Z, i, j;
	string  str;
        getline(fp, str);
        getline(fp, str);
        getline(fp, str);
	fp >> str >> str >> M >> str >> str >> Z;
	double	*number = new double[M * Z];
	for (i = 0;i < M;i++){
                for (j = 0;j < Z;j++)
                        fp >> number[i * Z + j];
        }
        fp.close();
	cout << "Read FMRI Data of shape:" << M << ' ' << Z << endl;
 //cout<<"Number 0:"<<number[0]<<endl;
 X_elements = M;
 Y_elements = Z;
 localSize = 31;
 
    // Number of total work items - localSize must be devisor
 globalSize = ceil((M*Z)/(float)localSize)*localSize;
cl_mem d_number;
cl_mem d_new1;
cl_mem d_new;
cl_mem d_G;
cl_mem d_mu;
cl_mem d_c;
cl_mem d_;
cl_mem d_cmax;
cl_mem d_I;
cl_mem d_beta;
cl_mem d_betaOLS;
cl_mem d_d;
cl_mem d_gamma;
cl_mem d_upper1;
cl_mem d_normb;
cl_mem d_ctrl;
cl_mem d_lVars;
cl_mem d_nVars;
cl_mem d_done;
cl_mem d_step;
cl_mem d_lasso;
cl_mem d_ind;

int n = min(M - 1, Z - 1);


   
 
    
 		d_number = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*M*Z, NULL, NULL);
      	d_new1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*(M-1)*Z, NULL, NULL);
       	d_new = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*(M-1)*Z, NULL, NULL);

  
     err = clEnqueueWriteBuffer(queue, d_number, CL_TRUE, 0,
                                   sizeof(double)*M*Z, number, 0, NULL, NULL);
     //memset(number,0,sizeof(double)*M*Z);
     //clEnqueueReadBuffer(queue, d_number, CL_TRUE, 0, sizeof(double)*M*Z, number, 0, NULL, NULL );
     //cout<<number[0]<<endl;
     
    double *set = new double[(M-1)*Z];
    double *number2 = new double[M*Z];
     memset(set,0,sizeof(double)*(M-1)*Z);
    
      err = clEnqueueWriteBuffer(queue, d_new, CL_TRUE, 0,
                                   sizeof(double)*(M-1)*Z, set, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_new1, CL_TRUE, 0,
                                   sizeof(double)*(M-1)*Z, set, 0, NULL, NULL);
/*
     clEnqueueReadBuffer(queue, d_number, CL_TRUE, 0,sizeof(double)*M*Z, number2, 0, NULL, NULL );

    cout<<"NUMBEr: "<<number2[0]<<endl; */
 	delete[] number2;
 	delete[] set;
 	
 	size_t local_item[2] = {31,31};
 	size_t global_item[2] = {((M+31-1)/31)*31,((Z+31-1)/31)*31};
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &d_number);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &d_new1);
    err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &d_new);
    err |= clSetKernelArg(kernel[0], 3, sizeof(int), &M);
    err |= clSetKernelArg(kernel[0], 4, sizeof(int), &Z);
 

    err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, global_item, local_item, 0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    double *h_new = new double[(M-1)*Z];
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_new, CL_TRUE, 0, (M-1)*Z*sizeof(double), h_new, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
    //double sum = 0;
   globalSize = ((Z+1000-1)/1000)*1000;
   localSize = 1000;

       		cout<<"d_trime: "<<h_new[34]<<"\t"<<h_new[35]<<"\t"<<h_new[36]<<endl;

    int M1 = M-1;
    //dProc<<<gz, bz>>>(d_new1, d_new, M - 1, Z);
    err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &d_new1);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &d_new);
    err |= clSetKernelArg(kernel[1], 2, sizeof( int), &M1);
    err |= clSetKernelArg(kernel[1], 3, sizeof( int), &Z);
     // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
   // memset(h_new,0,sizeof(double)*(M-1)*Z);
   
 //   clEnqueueReadBuffer(queue, d_new, CL_TRUE, 0, (M-1)*Z*sizeof(double), h_new, 0, NULL, NULL );
 	
    //Sum up vector c and print result divided by n, this should equal 1 within error
    //double sum = 0;
   
  // cout<<"h)new: "<<h_new[0]<<"\t"<<h_new[2]<<endl;
   
    //delete[] h_new;
    size_t dou = sizeof(double)*n*n;
   
   

	//cudaMalloc((void **)&d_G, n * n * sizeof(double));	
	 d_G = clCreateBuffer(context, CL_MEM_READ_WRITE, dou, NULL, NULL);
	
	//cudaMalloc((void **)&d_I, n * n * sizeof(double));
	d_I = clCreateBuffer(context, CL_MEM_READ_WRITE, dou, NULL, NULL);
	
	dou = sizeof(double)*(M-1)*N;
	
	//cudaMalloc((void **)&d_mu, (M - 1) * N * sizeof(double));
	d_mu = clCreateBuffer(context, CL_MEM_READ_WRITE, (M - 1) * N * sizeof(double), NULL, NULL);
	
	//cudaMalloc((void **)&d_d, (M - 1) * N * sizeof(double));
	d_d = clCreateBuffer(context, CL_MEM_READ_WRITE, (M - 1) * N * sizeof(double), NULL, NULL);
	
	dou = sizeof(double)*(Z-1)*N;			
	
	///cudaMalloc((void **)&d_c, (Z - 1) * N * sizeof(double));
	d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, (Z - 1) * N * sizeof(double), NULL, NULL);
	
	//cudaMalloc((void **)&d_, (Z - 1) * N * sizeof(double));
	d_ = clCreateBuffer(context, CL_MEM_READ_WRITE, (Z - 1) * N * sizeof(double), NULL, NULL);
	
	//cudaMalloc((void **)&d_beta, (Z - 1) * N * sizeof(double));
	d_beta = clCreateBuffer(context, CL_MEM_READ_WRITE, (Z - 1) * N * sizeof(double), NULL, NULL);
	
	dou = sizeof(double)*n*N;
	
	//cudaMalloc((void **)&d_betaOLS, n * N * sizeof(double));	
	d_betaOLS = clCreateBuffer(context, CL_MEM_READ_WRITE, (n) * N * sizeof(double), NULL, NULL);
	

	//cudaMalloc((void **)&d_gamma, n * N * sizeof(double));	
	d_gamma = clCreateBuffer(context, CL_MEM_READ_WRITE, n * N * sizeof(double), NULL, NULL);
	

	//cudaMalloc((void **)&d_lVars, n * N * sizeof(int));
	d_lVars = clCreateBuffer(context, CL_MEM_READ_WRITE, (n) * N * sizeof(int), NULL, NULL);
	
	dou = sizeof(double)*N;

	//cudaMalloc((void **)&d_cmax, N * sizeof(double));
	d_cmax = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(double), NULL, NULL);
	
	double *set2 = new double[N];


	//cudaMalloc((void **)&d_upper1, N * sizeof(double));
	d_upper1 = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(double), NULL, NULL);
	memset(set2,0,sizeof(double)*N);
	 err = clEnqueueWriteBuffer(queue, d_upper1, CL_TRUE, 0,
                                   sizeof(double)*N, set2, 0, NULL, NULL);
	
	//cudaMalloc((void **)&d_normb, N * sizeof(double));
	d_normb = clCreateBuffer(context, CL_MEM_READ_WRITE,  N * sizeof(double), NULL, NULL);
		memset(set2,0,sizeof(double)*N);
	 err = clEnqueueWriteBuffer(queue, d_normb, CL_TRUE, 0,
                                   sizeof(double)*N, set2, 0, NULL, NULL);
    
	dou = sizeof(int)*N;

		//cudaMalloc((void **)&d_nVars, N * sizeof(int));
	d_nVars = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, NULL);

	//cudaMalloc((void **)&d_step, N * sizeof(int));
	d_step = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, NULL);
	
	//cudaMalloc((void **)&d_ind, N * sizeof(int));
	d_ind = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, NULL);
	
	//cudaMalloc((void **)&d_done, N * sizeof(int));
	d_done = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, NULL);
	
	//cudaMalloc((void **)&d_lasso, N * sizeof(int));
	d_lasso = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, NULL);
	
	dou = sizeof(int);

	//cudaMalloc((void **)&d_ctrl, sizeof(int));
	d_ctrl = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, NULL);
	
	
 

cl_mem d_lambda;
cl_mem d_val;
d_lambda = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*N, NULL, NULL);
d_lambda = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*N, NULL, NULL);

	for(i = 0;i < Z;i += N){
		j = N;
		if(i + j > Z)j = Z - i;
	lars(d_new1, d_new, d_mu, d_c, d_, d_G, d_I, d_beta, d_betaOLS, d_d, d_gamma, d_cmax, d_upper1, d_normb, d_lVars,
	d_nVars, d_ind, d_step, d_done, d_lasso, d_ctrl, M, Z, i, j,err,kernel,queue, d_lambda,d_val);

	}
	

	//cudaFree(d_new1);
	 clReleaseMemObject(d_new1);
	//cudaFree(d_new);
	 clReleaseMemObject(d_new);
	//cudaFree(d_G);
	 clReleaseMemObject(d_G);
	//cudaFree(d_mu);
	 clReleaseMemObject(d_mu);
	//cudaFree(d_c);
	 clReleaseMemObject(d_c);
	//cudaFree(d_);
	 clReleaseMemObject(d_);
	//cudaFree(d_cmax);
	 clReleaseMemObject(d_cmax);
//	cudaFree(d_I);
	 clReleaseMemObject(d_I);
	//cudaFree(d_beta);
	 clReleaseMemObject(d_beta);
	//cudaFree(d_betaOLS);
	 clReleaseMemObject(d_betaOLS);
	//cudaFree(d_d);
	 clReleaseMemObject(d_d);
	//cudaFree(d_gamma);
	 clReleaseMemObject(d_gamma);
	//cudaFree(d_upper1);
	 clReleaseMemObject(d_upper1);
	//cudaFree(d_normb);
	 clReleaseMemObject(d_normb);
	//cudaFree(d_lVars);
	 clReleaseMemObject(d_lVars);
	//cudaFree(d_nVars);
	 clReleaseMemObject(d_nVars);
	//cudaFree(d_lasso);
	 clReleaseMemObject(d_lasso);
	//cudaFree(d_ind);
	 clReleaseMemObject(d_ind);
	//cudaFree(d_ctrl);
	 clReleaseMemObject(d_ctrl);
	//cudaFree(d_step);
	 clReleaseMemObject(d_step);
	//cudaFree(d_done);
	 clReleaseMemObject(d_done);










    clReleaseProgram(program);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseKernel(kernel[2]);
    clReleaseKernel(kernel[3]);
    clReleaseKernel(kernel[4]);
    clReleaseKernel(kernel[5]);
    clReleaseKernel(kernel[6]);
    clReleaseKernel(kernel[7]);
    clReleaseKernel(kernel[8]);
    clReleaseKernel(kernel[9]);
    clReleaseKernel(kernel[10]);
    clReleaseKernel(kernel[11]);
    clReleaseKernel(kernel[12]);
    clReleaseKernel(kernel[13]);
    clReleaseKernel(kernel[14]);
    clReleaseKernel(kernel[15]);
    clReleaseKernel(kernel[16]);
    clReleaseKernel(kernel[17]);
    clReleaseKernel(kernel[18]);
    clReleaseKernel(kernel[19]);
    clReleaseKernel(kernel[20]);
    clReleaseKernel(kernel[21]);
    clReleaseKernel(kernel[22]);
    clReleaseKernel(kernel[23]);
    clReleaseKernel(kernel[24]);
 	clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
 

	return 0;
}