/**
 * \file example_module.c
 * \brief Example of NEMEA module.
 * \author Vaclav Bartos <ibartosv@fit.vutbr.cz>
 * \author Marek Svepes <svepemar@fit.cvut.cz>
 * \date 2016
 */
/*
 * Copyright (C) 2013,2014,2015,2016 CESNET
 *
 * LICENSE TERMS
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name of the Company nor the names of its contributors
 *    may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * ALTERNATIVELY, provided that this notice is retained in full, this
 * product may be distributed under the terms of the GNU General Public
 * License (GPL) version 2 or later, in which case the provisions
 * of the GPL apply INSTEAD OF those given above.
 *
 * This software is provided ``as is'', and any express or implied
 * warranties, including, but not limited to, the implied warranties of
 * merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall the company or contributors be liable for any
 * direct, indirect, incidental, special, exemplary, or consequential
 * damages (including, but not limited to, procurement of substitute
 * goods or services; loss of use, data, or profits; or business
 * interruption) however caused and on any theory of liability, whether
 * in contract, strict liability, or tort (including negligence or
 * otherwise) arising in any way out of the use of this software, even
 * if advised of the possibility of such damage.
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <signal.h>
#include <getopt.h>
#include <libtrap/trap.h>
#include <unirec/unirec.h>
#include <stdlib.h>
#include "fields.h"

#define STRATEGY_RANDOM 0
#define STRATEGY_UNCERT 1

#define FLOW_ACCEPT 0
#define FLOW_REJECT 1

/**
 * Definition of fields used in unirec templates (for both input and output interfaces)
 */
UR_FIELDS (
   double* PROBA,
)

trap_module_info_t *module_info = NULL;


/**
 * Definition of basic module information - module name, module description, number of input and output interfaces
 */
#define MODULE_BASIC_INFO(BASIC) \
  BASIC("Stream sampler module", \
        "This module serves as stream sampler as a part of ALF.", 1, 1)
  //BASIC(char *, char *, int, int)


/**
 * Definition of module parameters - every parameter has short_opt, long_opt, description,
 * flag whether an argument is required or it is optional and argument type which is NULL
 * in case the parameter does not need argument.
 * Module parameter argument types: int8, int16, int32, int64, uint8, uint16, uint32, uint64, float, string
 */
#define MODULE_PARAMS(PARAM) \
  PARAM('s', "strategy", "Strategy for sampling. Expected integer 1-n, see help for more info.", required_argument, "int8") \
  PARAM('b', "budget", "Budget for sampling.", required_argument, "double") \
  PARAM('t', "threshold", "Threshold for PROBA value in UniRec.", optional_argument , "double") \
  PARAM('r', "rand", "Probability for random based sample strategies.", optional_argument , "double")


static int stop = 0;

TRAP_DEFAULT_SIGNAL_HANDLER(stop = 1)

int strategy;
double threshold ;
double budget = 0.1;
double probability;

int statistics[64] = {0};

/**
 * Return FLOW_ACCEPT if random number is lower than probability, else return FLOW_REJECT
 * 
 */
int strategy_random() {
   return ( (double) rand() / (double)RAND_MAX) < probability ? FLOW_ACCEPT : FLOW_REJECT;
}


int strategy_uncertainty_lc(double * proba, int proba_size) {
   double max = proba[0];
   int i, max_class = 0;
   for(int i = 1; i < proba_size; i++) {
      if (proba[i] > max) {
         max = proba[i];
         max_class = i;
      }
   }
   statistics[max_class]++;
   return max < threshold ? FLOW_ACCEPT : FLOW_REJECT;
}

int sample(int strategy, double * proba, int proba_size) {
   switch (strategy)
   {
   case STRATEGY_RANDOM:
      return strategy_random();
   case STRATEGY_UNCERT:
      return strategy_uncertainty_lc(proba, proba_size);
   default:
      return FLOW_REJECT;
   }
   return FLOW_REJECT;
}

int main(int argc, char **argv)
{
   int ret;
   signed char opt;

   /* **** TRAP initialization **** */
   /*
    * Macro allocates and initializes module_info structure according to MODULE_BASIC_INFO and MODULE_PARAMS
    * definitions on the lines 71 and 84 of this file. It also creates a string with short_opt letters for getopt
    * function called "module_getopt_string" and long_options field for getopt_long function in variable "long_options"
    */
   INIT_MODULE_INFO_STRUCT(MODULE_BASIC_INFO, MODULE_PARAMS)

   /*
    * Let TRAP library parse program arguments, extract its parameters and initialize module interfaces
    */
   TRAP_DEFAULT_INITIALIZATION(argc, argv, *module_info);

   /*
    * Register signal handler.
    */
   TRAP_REGISTER_DEFAULT_SIGNAL_HANDLER();

   /*
    * Parse program arguments defined by MODULE_PARAMS macro with getopt() function (getopt_long() if available)
    * This macro is defined in config.h file generated by configure script
    */
   while ((opt = TRAP_GETOPT(argc, argv, module_getopt_string, long_options)) != -1) {
      switch (opt) {
      case 't':
         threshold = atof(optarg);
         break;
      case 's':
         strategy = atoi(optarg);
         break;
      case 'r':
         probability = atof(optarg);
         break;
      case 'b':
         budget = atof(optarg);
         break;
      default:
         fprintf(stderr, "Invalid arguments.\n");
         FREE_MODULE_INFO_STRUCT(MODULE_BASIC_INFO, MODULE_PARAMS);
         TRAP_DEFAULT_FINALIZATION();
         return -1;
      }
   }

   /* **** Create UniRec templates **** */
   ur_template_t *in_tmplt = ur_create_input_template(0, "PROBA", NULL);
   if (in_tmplt == NULL){
      fprintf(stderr, "Error: Input template could not be created.\n");
      return -1;
   }
   /* **** Main processing loop **** */
   // Read data from input, decide and send to output.

   const void * data = NULL;
   uint16_t in_rec_size;

   while (!stop) {
         ret = trap_recv(0, &data, &in_rec_size);
      if (ret == TRAP_E_OK || ret == TRAP_E_FORMAT_CHANGED) {
         if (ret == TRAP_E_OK) {
            if (in_rec_size <= 1) {
               stop = 1;
            }
         } else {
            // Get the data format of senders output interface (the data format of the output interface it is connected to)
            const char *spec = NULL;
            uint8_t data_fmt = TRAP_FMT_UNKNOWN;
            if (trap_get_data_fmt(TRAPIFC_INPUT, 0, &data_fmt, &spec) != TRAP_E_OK) {
               fprintf(stderr, "Data format was not loaded.");
               return;
            }
            // Set the same data format to repeaters output interface
            trap_set_data_fmt(0, TRAP_FMT_UNIREC, spec);
         }

         if (stop == 1){
            break;
         } else {
            ret = trap_send(0, data, in_rec_size);
            if (ret == TRAP_E_OK) {
               continue;
            }
            TRAP_DEFAULT_SEND_DATA_ERROR_HANDLING(ret, continue, break)
         }
      } else {
         TRAP_DEFAULT_GET_DATA_ERROR_HANDLING(ret, continue, break)
      }
   }
   




   /* **** Cleanup **** */

   // Do all necessary cleanup in libtrap before exiting
   TRAP_DEFAULT_FINALIZATION();

   // Release allocated memory for module_info structure
   FREE_MODULE_INFO_STRUCT(MODULE_BASIC_INFO, MODULE_PARAMS)

   // Free unirec templates and output record
   ur_free_template(in_tmplt);
   ur_finalize();

   return 0;
}

