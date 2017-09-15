/********************
Author: Erik Talvitie
Date: 2016
********************/

#ifndef CONVOLUTIONAL_BINARY_CTS
#define CONVOLUTIONAL_BINARY_CTS

#include "cts.hpp"
#include "common.hpp"
#include "SamplingModel.h"

#include <vector>
#include <boost/tuple/tuple.hpp>
//#include <tuple>

using namespace std;
using namespace boost;

/*A model for k-order MDPs with image observations
It applies the CTS algorithm at each position, based on
the pixels in a neighborhood around that position in the
previous k observation*/
class ConvolutionalBinaryCTS : public SamplingModel<int>
{
  private:
   int width;                 //Width of the images
   int height;                //Height of the images
   int neighborhoodWidth;     //Width of the convolutional window
   int neighborhoodHeight;    //Height of the convolutional window
   int bitsPerAction;         //How many bits are needed to encode an action
   int bitsPerPixel;          //How many bits are needed to encode a color
   int numColors;             //How many colors are possible
   int order;                 //The order of the MDP

   //Three CTS models will be used...
   //This one is for predicting pixels (data is shared across positions)
   mutable SwitchingTree* ct;
   //This one is used for predicting rewards
   mutable SwitchingTree* rct;
   //This one is used for predicting the end of the episode
   mutable SwitchingTree* ect;

   //Random number generation
   randsrc_t rng;
   randgen_t internal_uniform;
   randgen_t& uniform;

   //For saving and restoring the state
   int savedHistoryLength;
   int savedNumTraj;

   //The history
   vector<vector<int> > actHistory;
   vector<vector<vector<int> > > obsHistory;
   vector<vector<bool> > rHistory;
   vector<vector<bool> > endHistory;

   //Used to pre-calculate neighborhood offsets (just runtime optimization)
   vector<vector<int> > offsetToEncodedPos;

   //Encodes the neighborhood around the given position
   //into an input vector for the CTS model
   void encode(const vector<int>& obs, int pos, vector<bit_t>& encoded) const;
   void encode(const vector<int>& obs, int x, int y, vector<bit_t>& encoded) const;   

   //Encodes the entire observation
   //(for reward and end prediction)
   void encode(const vector<int>& obs, vector<bit_t>& encoded) const;

   //Encodes the action
   void encode(int act, vector<bit_t>& encoded) const;

   //Samples the next observation/reward/end from the given trajectory and step
   void sample(int traj, int step, int act, vector<int>& sampled, bool& reward, bool& endTraj);

   //Updates the models with the history starting at step
   //and going for numUpdates steps
   void updateActObs(int traj, int step, int numUpdates);
   void updateREnd(int traj, int step, int numUpdates);

   //Sets up the context for a given position at the given time step
   //(Encodes the relevant action and neighborhood, and updates
   //the histories in the models)
   void setUpContext(int pos, int traj, int step) const;
   //Sets up the context for the reward and end models
   void setUpContext(int traj, int step) const;

   //Initialize the model
   void init(int neighborhoodWidth, int neighborhoodHeight, int numActions, int numColors);

  public:
   ConvolutionalBinaryCTS(int width, int height, int neighborhoodWidth, int neighborhoodHeight, int numActions, int order, int seed);
   ConvolutionalBinaryCTS(int width, int height, int neighborhoodWidth, int neighborhoodHeight, int numActions, int order, randgen_t& uniform);
   ~ConvolutionalBinaryCTS();

   //Update the model with a new step (maybe learn from it)
   void update(int act, const vector<int>& obs, bool reward, bool endTraj, bool learn = true);
   void update(int act, const vector<int>& obs, int reward, bool endTraj);

   void sample(int act, vector<int>& sampled, bool& reward, bool& endTraj);

   //Give the probability of the observation
   //given the action and the model's current state
   double predict(int act, const vector<int>& obs, bool print=false) const;
   //Give the probability of the reward
   //given the action and the model's current state
   double predictR(int act, const vector<int>& obs, int reward) const;
   //Give the probability of the trajectory end value
   //given the action and the model's current state
   double predictEnd(int act, const vector<int>& obs, bool end) const;

   //Samples a next state and then updates using that state
   void takeAction(int act, vector<int>& obs, int& reward, bool& endEpisode);
   void takeAction(int act, int& reward, bool& endEpisode);

   //Resets to the initial state (i.e. begins a new, empty trajectory)
   void reset();

   //Trains the model using a batch of obs, action, next obs triples
   void batchUpdate(const vector<tuple<vector<vector<int> >, vector<int>, int, vector<int>, int, bool> >& dataset); //obs context, action context, nextAct, nextObs, reward, endEpisode

   //Save the state for future retrieval
   void saveState();
   //Retrieve the saved state
   void retrieveState();
};

#endif
