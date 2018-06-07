/********************
Author: Erik Talvitie
********************/

#ifndef SHOOTER_MODEL
#define SHOOTER_MODEL

#include "SamplingModel.h"

class ShooterModel : public SamplingModel<int>
{
  private:
   int width;
   int height;
   bool movingSweetSpot;

   int shipPos;
   vector<int> targets;
   int targetPhase;
   vector<pair<int, int> > bullets;

   vector<int> lastObs;

   int savedShipPos;
   vector<int> savedTargets;
   int savedTargetPhase;
   vector<pair<int, int> > savedBullets;

   vector<int> savedLastObs;

  public:
   ShooterModel(int numTargets, int height, bool movingSweetSpot = false);

   //Take the given action in the model
   //Should sample a transation and update the state accordingly
   //Fills in obs with the sampled observation

   //Always assigns 0 to reward and false to endEpisode
   //(In these experiments, the reward function is known and episodes have
   //fixed length so it is better to get these values externally).
   void takeAction(int action, vector<int>& obs, int& reward, bool& endEpisode);

   //Reset the game to its initial state
   void reset();

   //Save the games's state for later retrieval
   void saveState();
   //Reset the game to the saved state
   void retrieveState();
};

#endif
