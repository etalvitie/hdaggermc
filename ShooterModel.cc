/********************
Author: Erik Talvitie
Date: 2016
********************/

#include "ShooterModel.h"
#include <iostream>
#include <cstdlib>

ShooterModel::ShooterModel(int numTargets, int height, bool movingSweetSpot) :
   SamplingModel<int>(4, height*numTargets*5),
   width(numTargets*5),
   height(height),
   movingSweetSpot(movingSweetSpot),
   targets(numTargets, 0)
{
   reset();
}

void ShooterModel::reset()
{
   shipPos = 0;
   targetPhase = 1;
   if(movingSweetSpot)
   {
      targetPhase = 0;
   }

   for(unsigned i = 0; i < targets.size(); i++)
   {
      targets[i] = 1;
   }

   bullets.clear();
}

void ShooterModel::takeAction(int act, vector<int>& obs, int& reward, bool& endEpisode)
{
   //We're not going to do anything with these
   //(Let them be sorted out externally)
   reward = 0;
   endEpisode = false;

   obs.resize(this->numDim);
   fill(obs.begin(), obs.end(), 0);

   //Clear away any targets that exploded in the last step
   for(unsigned i = 0; i < targets.size(); i++)
   {
      if(targets[i] > 1)
      {
	 targets[i] = 0;
      }
   }

   //If the sweet spots move, move them
   
   if(movingSweetSpot)
   {
      targetPhase = (targetPhase + 1)%4;
   }
   int sweetSpot = targetPhase;
   if(targetPhase%2)
   {
      sweetSpot = 1;
   }
   /*
   //Trying out more complicated movement
   if(movingSweetSpot)
   {
      targetPhase = (targetPhase + 1)%8;
   }
   int sweetSpot = targetPhase/2;
   if(targetPhase > 5)
   {
      sweetSpot = 1;
   }
   */
   //Update the bullets...
   vector<int> toErase;
   for(unsigned i = 0; i < bullets.size(); i++)
   {
      if(bullets[i].second > 0) //Mostly bullets just move up
      {
	 bullets[i].second--;
      }
      else //If a bullet reaches the top of the screen, destroy it
      {
	 toErase.push_back(i); 
      }

      //If a bullet is at the bottom of the targets..
      if(bullets[i].second == 4)
      {
	 //Get the target the bullet is near
	 int targetIndex = bullets[i].first/5;
	 //Get the position within the target
	 int intraTargetPos = bullets[i].first%5;
	 //If the bullet is hitting a target...
	 if(intraTargetPos > 0 && intraTargetPos < 4 && targets[targetIndex] == 1)
	 {
	    targets[targetIndex] = 2; //Make the target explode
	    if(intraTargetPos == sweetSpot+1)
	    {
	       targets[targetIndex] = 3; //Bullseye! -> Special explosion
	    }
	    toErase.push_back(i); //Destroy the bullet
	 }
      }
   }

   //Get rid of all the destroyed bullets
   for(unsigned i = 0; i < toErase.size(); i++)
   {
      bullets[toErase[i]] = bullets.back();
      bullets.pop_back();
   }

   //Now move the ship according to the action
   //act == 0: no-op
   if(act == 1) //left
   {
      if(shipPos > 0)
      {
	 shipPos--;
      }
   }
   else if(act == 2) //right
   {
      if(shipPos < width - 3)
      {
	 shipPos++;
      }
   }
   else if(act == 3) //shoot
   {
      if(bullets.empty() || bullets.back().second < height - 5) //Don't shoot too fast
      {
	 bullets.push_back(make_pair(shipPos+1, height - 3)); //Create a bullet
      }
   }
   
   //Time to fill in the observation!
   
   //Now draw the targets   
   for(unsigned i = 0; i < targets.size(); i++)
   {
      if(targets[i] == 1) //Draw the target
      {
	 int corner = 2*width + i*5 + 1;
	 for(unsigned y = 0; y < 3; y++)
	 {
	    for(unsigned x = 0; x < 3; x++)
	    {
	       obs[corner + width*y + x] = 1;
	    }
	 }

	 obs[corner+width+sweetSpot] = 0;
      }
      else if(targets[i] == 2) //Draw the explosion
      {
	 int corner = 2*width + i*5 + 1;
	 for(unsigned y = 0; y < 3; y+=2)
	 {
	    for(unsigned x = 0; x < 3; x+=2)
	    {
	       obs[corner + width*y + x] = 1;
	    }
	 }
	 obs[corner + width + 1] = 1;
      }
      else if(targets[i] == 3) //Draw the special explosion
      {
	 int corner = 2*width + i*5 + 1;
	 for(unsigned y = 0; y < 3; y+=2)
	 {
	    obs[corner + width*y + 1] = 1;
	 }
	 for(unsigned x = 0; x < 3; x+=2)
	 {
	    obs[corner + width + x] = 1;
	 }
      }
   }

   //Now draw the bullets
   for(unsigned i = 0; i < bullets.size(); i++)
   {
      obs[bullets[i].second*width + bullets[i].first] = 1;
   }

   //Now draw the ship
   for(int x = 0; x < 3; x++)
   {
      obs[(height - 1)*width + shipPos + x] = 1;
      obs[(height - 2)*width + shipPos + 1] = 1;
   }

   lastObs = obs;
}

void ShooterModel::saveState()
{
   savedShipPos = shipPos;
   savedTargets = targets;
   savedTargetPhase = targetPhase;
   savedBullets = bullets;

   savedLastObs = lastObs;
}

void ShooterModel::retrieveState()
{
   shipPos = savedShipPos;
   targets = savedTargets;
   targetPhase = savedTargetPhase;
   bullets = savedBullets;

   lastObs = savedLastObs;
}
