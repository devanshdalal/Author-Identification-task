/*
 *  Anthony Pasqualoni
 *  Independent Study: Neural Networks and Pattern Recognition
 *  Adviser: Dr. Hrvoje Podnar, SCSU
 *  June 27, 2006
 * 
 *  MLP with 2 hidden layers for identifying authors of English sonnets
 *  Input data for training and testing provided by extract.py and saved in features.txt
 *
 *  Code based on algorithms described in Neural Networks: A Comprehensive Foundation,
 *  by Simon Haykin.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// MALLOC definition from Nelson Castillo, www.geocities.com/arhuaco/log/src/cagraph.c:
#define MALLOC(var,size) \
        (var) = malloc(size); \
           if(!(var)) \
	      fprintf(stderr, "%s:%d: out of memory\n", __FILE__, __LINE__), \
	      exit(EXIT_FAILURE); \

// neural network parameters:
#define INPUT_NODE_AMT 6       // amount of feature measurements per sonnet 
#define HIDDEN_NODE_AMT 20     // amount of nodes in hidden layers
#define OUTPUT_NODE_AMT 1      // output:  +1 = shakespeare, -1 = other author

#define MAX_ITERATIONS 10000   // maximum iterations for training
#define PATTERN_AMT 200        // amount of sonnets from which features have been extracted
                               // half are for training, half for testing
#define VERBOSE 0              // set to true for verbose output

#if (RAND_MAX < 2147483647)
   #undef RAND_MAX
   #define RAND_MAX 2147483647
#endif

// neural network structure:
typedef struct mlp {

   // node amounts:
   unsigned int input_node_amt;
   unsigned int hidden_node_amt;
   unsigned int output_node_amt;

   // input nodes:
   double x[INPUT_NODE_AMT];

   // weights and biases include last 3 values in case alpha is used:
   double w1[HIDDEN_NODE_AMT][INPUT_NODE_AMT]  [3]; 
   double w2[HIDDEN_NODE_AMT][HIDDEN_NODE_AMT] [3]; 
   double w3[OUTPUT_NODE_AMT][HIDDEN_NODE_AMT] [3]; 
   double bias1[HIDDEN_NODE_AMT] [3];   
   double bias2[HIDDEN_NODE_AMT] [3];   
   double bias3[OUTPUT_NODE_AMT] [3];   

   // node outputs:
   double v1[HIDDEN_NODE_AMT];
   double v2[HIDDEN_NODE_AMT];
   double v3[OUTPUT_NODE_AMT];
   double y1[HIDDEN_NODE_AMT];
   double y2[HIDDEN_NODE_AMT];
   double y3[OUTPUT_NODE_AMT];

   // back propagation:
   double delta1[HIDDEN_NODE_AMT];
   double delta2[HIDDEN_NODE_AMT];
   double delta3[OUTPUT_NODE_AMT];

   // desired output:
   double d[OUTPUT_NODE_AMT];                            

   // error:
   double e[OUTPUT_NODE_AMT];                            

} MLP;

// global variables:

struct mlp nn; // neural network

unsigned int iteration; // current iteration
double training_set[PATTERN_AMT][INPUT_NODE_AMT + 1]; // training data from both author groups

double n,n0;       // learning rate and first value of learning rate
double alpha;      // experimental; set to 0 for best results
double a,b;        // constants for calculating phi_prime

// function definitions:
unsigned int test (unsigned int pattern_amt);
void network_init (void);
void load_training_sets (char * fname);
void set_inputs (double * training_set_index);
void set_d (double * training_set_index);
void initialize_weights (void);
double sigmoid (double x);
void define_y1 (void);
void define_y2 (void);
void define_y3 (void);
void define_error (void);
void adjust_output_weights();
void adjust_y2_weights();
void adjust_y1_weights();
void print_y3 (void);
void print_x (void);

int main (void) {

   int i,j;
   unsigned int rand_seed;        // seed value for random number generator
   unsigned int pattern;
   unsigned int training_failures;
   unsigned int failures;
   unsigned int run, runs;
   double fail_avg;

   // initialize random number generator, use constant seed for repeatable results:
   rand_seed = 1;       
   srandom(rand_seed);

   runs = 10;
   fail_avg = 0;

   // train and test network:
   for (run = 0; run < runs; run++) {

      load_training_sets ("features.txt");
   
      if (VERBOSE)
         printf("Initializing network:\n");
      network_init();
        
      if (run == 0) {
 	 printf("\nHidden layers:      2\nNodes per layer:    %d\nMaximum iterations: %d\nLearning rate:      %f\n\n",
            HIDDEN_NODE_AMT,MAX_ITERATIONS,n);
         printf ("Input data from 200 sonnets:\n");

         printf ("   Training: 50 sonnets by Shakespeare; 50 by Spenser, Smith, & Griffin\n");
         printf ("   Test:     50 sonnets by Shakespeare; 50 by Spenser, Smith, & Griffin\n");

         printf("\nRun:     Iterations:    Training failures:    Test failures:\n");
      }

      // train network:
      failures = 100;
      for (iteration = 0; iteration < (MAX_ITERATIONS) && (failures > 0); iteration++) {

         pattern = random() % (PATTERN_AMT/2);
   
         set_inputs (training_set[pattern]);
         set_d (training_set[pattern]);

         define_y1();
         define_y2();
         define_y3();

         if ( !( iteration % 1000) && VERBOSE) 
            printf("\nIteration: %u:\n",iteration);

         adjust_output_weights();
         adjust_y2_weights();
         adjust_y1_weights();
   
         if ( !( iteration % 1000) ) {
            failures = test (PATTERN_AMT/2);
            if (VERBOSE)
               printf("failures at iteration %u: %u\n",iteration,failures);
         }
   
      } // for iteration 
   
      // test network:
      if (VERBOSE)
         printf("final test:\n");

      training_failures = test (PATTERN_AMT/2);
      failures = test (PATTERN_AMT);
   
      printf(" %2d      %10d            %3u of %3d        %3u of %3d\n",
         run + 1,iteration,training_failures,PATTERN_AMT/2,failures - training_failures,PATTERN_AMT/2);
      fail_avg += failures - training_failures;

    } // for run

    fail_avg = fail_avg/runs;
    printf("\nAverage accuracy in identifying author after training: %3.2f\n\n", 100.0 - fail_avg );

  return (0);

}

unsigned int test (unsigned int pattern_amt) {
// test network

   int i;
   unsigned int failures;
   unsigned int pattern;

   if (VERBOSE)
      printf("=========================== test ================================\n\n");

   failures = 0;
   for (pattern = 0; pattern < pattern_amt; pattern++) {
      set_inputs (training_set[pattern]);

      if (VERBOSE)
         printf("\npattern: %u desired output:%f \n",pattern, training_set[pattern][INPUT_NODE_AMT]);

      define_y1();
      define_y2();
      define_y3();

      if (VERBOSE)
         print_y3();

      if (OUTPUT_NODE_AMT == 1) {
         if ( (nn.y3[0] > 0) && (training_set[pattern][INPUT_NODE_AMT] == -1.0 ) ) {
            failures++;
            if (VERBOSE)
               printf("failure at node 0.\n");
         }
         if ( (nn.y3[0] < 0) && (training_set[pattern][INPUT_NODE_AMT] == 1.0 ) ) {
            failures++;
            if (VERBOSE)
               printf("failure at node 0.\n");
         }
      }
      else {
         for (i = 0; i < nn.output_node_amt; i++) {
            if ( (nn.y3[i] > 0) && (i != training_set[pattern][INPUT_NODE_AMT]) ) {
               failures++;
               if (VERBOSE)
                  printf("failure at node %d.\n",i);
            }
            if ( (i == training_set[pattern][INPUT_NODE_AMT]) && (nn.y3[i] <= 0) ) {
               failures++;
               if (VERBOSE)
                  printf("failure at node %d.\n",i);
            }
         } // for i
      } // else

   } // for pattern

   return (failures);

}

void load_training_sets (char * fname) {

   int i,j;
   FILE * fp;
   int pattern;
   
   fp = fopen (fname,"r");
   printf("%s", fname );
   assert (fp);

   for (pattern = 0; pattern < PATTERN_AMT; pattern++) {
      for (i = 0; i < (INPUT_NODE_AMT); i++) {
         assert ( fscanf(fp,"%lf,",&training_set[pattern][i]) );
      }
      // read desired output:
      assert ( fscanf(fp,"%lf,",&training_set[pattern][INPUT_NODE_AMT]) );
      // read terminating string:
      assert ( fscanf(fp,"%d,",&i) );
      assert ( i == 1000 );
   }

   fclose (fp);

}

void network_init (void) {

   nn.input_node_amt = INPUT_NODE_AMT;
   nn.hidden_node_amt = HIDDEN_NODE_AMT;
   nn.output_node_amt = OUTPUT_NODE_AMT;

   initialize_weights();

   // values for a,b from Neural Networks: A Comprehensive Foundation:
   a = 1.7159;
   b = 0.66667;

   alpha = 0;

   n = 0.05;

   n0 = n;

}

void set_inputs (double * training_set_index) {

   int i,j;

   for (i = 0; i < nn.input_node_amt; i++)
      nn.x[i] = training_set_index[i];

}

void set_d (double * training_set_index) {

   int i,j;

   if (OUTPUT_NODE_AMT == 1) {
      nn.d[0] = training_set_index[INPUT_NODE_AMT];
      return;
   }

   for (i = 0; i < nn.output_node_amt; i++) 
      if (i == (int)training_set_index[INPUT_NODE_AMT] ) {
         nn.d[i] = 1.0;
      }
      else
         nn.d[i] = -1.0;
}

void initialize_weights (void) {

   int i,j;

   for (j = 0; j < nn.hidden_node_amt; j++) {

      // first hidden layer:
      nn.bias1[j][0] = (double)random()/RAND_MAX;
      nn.bias1[j][0] = nn.bias1[j][0] * 2.0 - 1.0;
      assert (nn.bias1[j][0] < 1.0);

      // second hidden layer:
      nn.bias2[j][0] = (double)random()/RAND_MAX;
      nn.bias2[j][0] = nn.bias2[j][0] * 2.0 - 1.0;
      assert (nn.bias2[j][0] < 1.0);

      // first hidden layer:
      for (i = 0; i < nn.input_node_amt; i++) {
         nn.w1[j][i][0] = (double)random()/RAND_MAX;
         nn.w1[j][i][0] = nn.w1[j][i][0] * 2.0 - 1.0;
         assert (nn.w1[j][i][0] < 1.0);
      }

      // second hidden layer:
      for (i = 0; i < nn.hidden_node_amt; i++) {
         nn.w2[j][i][0] = (double)random()/RAND_MAX;
         nn.w2[j][i][0] = nn.w2[j][i][0] * 2.0 - 1.0;
         assert (nn.w2[j][i][0] < 1.0);
      }

   }

   // output nodes:
   for (j = 0; j < nn.output_node_amt; j++) {
      nn.bias3[j][0] = (double)random()/RAND_MAX;
      nn.bias3[j][0] = nn.bias3[j][0] * 2.0 - 1.0;
     for (i = 0; i < nn.hidden_node_amt; i++) {
         nn.w3[j][i][0] = (double)random()/RAND_MAX;
         nn.w3[j][i][0] = nn.w3[j][i][0] * 2.0 - 1.0;
      }
   }

}

void define_y1 (void) {

   int i,j;

   // determine net input:
   for (j = 0; j < nn.hidden_node_amt; j++) {
      // add bias:
      nn.v1[j] = nn.bias1[j][iteration % 3] * 1;
      for (i = 0; i < nn.input_node_amt; i++) 
         nn.v1[j] += nn.w1[j][i][iteration % 3] * nn.x[i];
   }

   // apply sigmoid function:
   for (j = 0; j < nn.hidden_node_amt; j++)
      nn.y1[j] = sigmoid (nn.v1[j]);

}

void define_y2 (void) {

   int i,j;

   // determine net input:
   for (j = 0; j < nn.hidden_node_amt; j++) {
      // add bias:
      nn.v2[j] = nn.bias2[j][iteration % 3] * 1;
      for (i = 0; i < nn.hidden_node_amt; i++) 
         nn.v2[j] += nn.w2[j][i][iteration % 3] * nn.y1[i];
   }

   // apply sigmoid function:
   for (j = 0; j < nn.hidden_node_amt; j++)
      nn.y2[j] = sigmoid (nn.v2[j]);

}

void define_y3 (void) {

   int i,j;

   // determine net input:
   for (j = 0; j < nn.output_node_amt; j++) {
      // add bias:
      nn.v3[j] = nn.bias3[j][iteration % 3] * 1;
      for (i = 0; i < nn.hidden_node_amt; i++) 
         nn.v3[j] += nn.w3[j][i][iteration % 3] * nn.y2[i];
   }

   // apply sigmoid function:
   for (j = 0; j < nn.output_node_amt; j++)
      nn.y3[j] = sigmoid (nn.v3[j]);

}

double sigmoid (double x) {

   return (a * tanh (b * x));  

}

void define_error (void) {

   int i,j;

   for (j = 0; j < nn.output_node_amt; j++) 
      nn.e[j] = nn.d[j] - nn.y3[j];

}

void adjust_output_weights (void) {

   int i,j;

   double phi_prime;

   define_error();

   for (j = 0; j < nn.output_node_amt; j++) {

      // phi_prime calculation from Neural Networks: A Comprehensive Foundation, p. 169:
      phi_prime = (b / a) * (a - nn.y3[j]) * (a + nn.y3[j]);

      nn.delta3[j] = nn.e[j] * phi_prime;

     // adjust bias:
     if (iteration > 0) 
        nn.bias3[j][(iteration + 1) % 3] = 
            nn.bias3[j][iteration % 3] + alpha * nn.bias3[j][(iteration - 1) % 3] + n * nn.delta3[j] * 1;
      else            
      nn.bias3[j][(iteration + 1) % 3] = 
         nn.bias3[j][iteration % 3] + n * nn.delta3[j] * 1;

      // adjust weights:
      for (i = 0; i < nn.hidden_node_amt; i++) {
         if (iteration > 0) 
            nn.w3[j][i][(iteration + 1) % 3] = 
               nn.w3[j][i][iteration % 3] + alpha * nn.w3[j][i][(iteration - 1) % 3] + n * nn.delta3[j] * nn.y2[i];
         else            
            nn.w3[j][i][(iteration + 1) % 3] = 
               nn.w3[j][i][iteration % 3] + n * nn.delta3[j] * nn.y2[i];
       }
   }

} // adjust_output_weights

void adjust_y2_weights (void) {

   int i,j,k;
   double phi_prime;
   double sum;

   for (j = 0; j < nn.hidden_node_amt; j++) {

      // phi_prime calculation from Neural Networks: A Comprehensive Foundation, p. 169:
      phi_prime = (b / a) * (a - nn.y2[j]) * (a + nn.y2[j]);  

      sum = 0;
      for (k = 0; k < nn.output_node_amt; k++) {
         sum += nn.delta3[k] * nn.w3[k][j][iteration % 3];
      }

      nn.delta2[j] = phi_prime * sum;

     // adjust bias:
     if (iteration > 0) 
         nn.bias2[j][(iteration + 1) % 3] = 
            nn.bias2[j][iteration % 3] + alpha * nn.bias2[j][(iteration - 1) % 3] + n * nn.delta2[j] * 1;
     else            
         nn.bias2[j][(iteration + 1) % 3] = 
            nn.bias2[j][iteration % 3] + n * nn.delta2[j] * 1;

     // adjust weights:
     for (i = 0; i < nn.hidden_node_amt; i++) {
 	 if (iteration > 0) 
            nn.w2[j][i][(iteration + 1) % 3] = 
               nn.w2[j][i][iteration % 3] + alpha * nn.w2[j][i][(iteration - 1) % 3] + n * nn.delta2[j] * nn.y1[i];
         else
            nn.w2[j][i][(iteration + 1) % 3] = 
               nn.w2[j][i][iteration % 3] + n * nn.delta2[j] * nn.y1[i];
     }
  } // for j

} // adjust_y2_weights

void adjust_y1_weights (void) {

   int i,j,k;
   double phi_prime;
   double sum;

   for (j = 0; j < nn.hidden_node_amt; j++) {

      // Neural Networks: A Comprehensive Foundation, p. 169:
      phi_prime = (b / a) * (a - nn.y1[j]) * (a + nn.y1[j]);  

      sum = 0;
      for (k = 0; k < nn.hidden_node_amt; k++) {
         sum += nn.delta2[k] * nn.w2[k][j][iteration % 3];
      }

      nn.delta1[j] = phi_prime * sum;

     // adjust bias:
     if (iteration > 0) 
         nn.bias1[j][(iteration + 1) % 3] = 
            nn.bias1[j][iteration % 3] + alpha * nn.bias1[j][(iteration - 1) % 3] + n * nn.delta1[j] * 1;
     else            
         nn.bias1[j][(iteration + 1) % 3] = 
            nn.bias1[j][iteration % 3] + n * nn.delta1[j] * 1;

     // adjust weights:
     for (i = 0; i < nn.input_node_amt; i++) {
 	 if (iteration > 0) 
            nn.w1[j][i][(iteration + 1) % 3] = 
               nn.w1[j][i][iteration % 3] + alpha * nn.w1[j][i][(iteration - 1) % 3] + n * nn.delta1[j] * nn.x[i];
         else
            nn.w1[j][i][(iteration + 1) % 3] = 
               nn.w1[j][i][iteration % 3] + n * nn.delta1[j] * nn.x[i];
       }
   } // for j

} // adjust_y1_weights

void print_y3 (void) {

   int i,j;

   printf("output nodes:\n");
   for (j = 0; j < nn.output_node_amt; j++) 
      printf("%d: %f\n",j,nn.y3[j]);


} // print_y3

void print_x (void) {

   int i,j;

   printf("input nodes:\n");
   for (j = 0; j < nn.input_node_amt; j++) 
      printf("%d: %f\n",j,nn.x[j]);

} // print_x

