#pragma once
#include "DomeImageManager.h"

namespace Module_BodyPose
{
	bool Load_Undist_PoseDetectMultipleCamResult_MultiPoseMachine_19jointFull(const char* poseDetectFolder,const char* poseDetectSaveFolder,const int currentFrame, CDomeImageManager& domeImMan,bool isHD);
}