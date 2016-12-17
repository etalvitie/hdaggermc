/********************
Author: Erik Talvitie
Date: 2016
********************/

#ifndef SAMPLING_MODEL
#define SAMPLING_MODEL

#include <vector>
#include <set>

using namespace std;

/*An abstract class template for a forward sampling model*/
template <class actObs_t>
class SamplingModel
{
  protected:
   int numActs;
   int numDim;
   
  public:
   SamplingModel(int numActs, int numDim);
   virtual ~SamplingModel(){}

   //Take the given action in the model
   //Should sample a transation and update the state accordingly
   //Fills in obs, reward, and endEpisode with the sampled values
   virtual void takeAction(actObs_t act, vector<actObs_t>& obs, int& reward, bool& endEpisode) = 0;

   //Reset the model to its initial state
   virtual void reset() = 0;
   
   //Save the model's state for later retrieval
   virtual void saveState() = 0;
   //Reset the model to the saved state
   virtual void retrieveState() = 0;

   virtual int getNumActs();
   virtual int getObsDim();
};

template <class actObs_t>
SamplingModel<actObs_t>::SamplingModel(int numActs, int numDim) :
   numActs(numActs),
   numDim(numDim)
{
}

template <class actObs_t>
int SamplingModel<actObs_t>::getNumActs()
{
   return numActs;
}

template <class actObs_t>
int SamplingModel<actObs_t>::getObsDim()
{
   return numDim;
}

#endif
