#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <conio.h>
#include <string>

#include <pylon/PylonIncludes.h>
//#include <pylon/usb/BaslerUsbCamera.h>
#include <pylon/PylonIncludes.h>
#include <pylon/gige/BaslerGigECamera.h>
#include <pylon/gige/BaslerGigEInstantCamera.h>
#include <pylon/gige/_BaslerGigECameraParams.h>
#include <cstdlib>
#include <vector>
#include <numeric>
#include<windows.h>
#include <time.h> 
//typedef Pylon::CBaslerUsbCamera Camera_u;
typedef Pylon::CBaslerGigECamera Camera_t;
//typedef Pylon::CBaslerGigEInstantCamera Camera_t;
#ifdef PYLON_WIN_BUILD
#include <pylon/PylonGUI.h>    
#endif
//using namespace Basler_UsbCameraParams;
//using namespace Basler_UsbStreamParams;
using namespace Basler_GigECameraParams;
using namespace Basler_GigEStreamParams;

using namespace std;
using namespace cv;
using namespace Pylon;
using namespace GenApi;



string savepath = "D://APP//pylonpic//";
string savepath0;
string savepath_1;
string savepath_2;

int height = 1200;
int width = 1600;
int OffsetX = 0;
int OffsetY = 0;
int Exp = 50000;

Mat src = Mat::zeros(height, width, CV_8UC1);
Mat src1 = Mat::zeros(height, width, CV_8UC1);

class CHeartbeatHelper
{
public:
	explicit CHeartbeatHelper(CBaslerGigECamera& Camera)
		: m_pHeartbeatTimeout(NULL)
	{
		// m_pHeartbeatTimeout may be NULL
		//m_pHeartbeatTimeout = camera.GetTLNodeMap().GetNode("HeartbeatTimeout");
		m_pHeartbeatTimeout = Camera.GetTLNodeMap()->GetNode("HeartbeatTimeout");
	}

	bool SetValue(int64_t NewValue)
	{
		// Do nothing if no heartbeat feature is available.
		if (!m_pHeartbeatTimeout.IsValid())
			return false;

		// Apply the increment and cut off invalid values if neccessary.
		int64_t correctedValue = NewValue - (NewValue % m_pHeartbeatTimeout->GetInc());

		m_pHeartbeatTimeout->SetValue(correctedValue);
		return true;
	}

	bool SetMax()
	{
		// Do nothing if no heartbeat feature is available.
		if (!m_pHeartbeatTimeout.IsValid())
			return false;

		int64_t maxVal = m_pHeartbeatTimeout->GetMax();
		return SetValue(maxVal);
	}

protected:
	GenApi::CIntegerPtr m_pHeartbeatTimeout; // Pointer to the node, will be NULL if no node exists.
};



int newfold(string folderPath)
{
	bool flag = CreateDirectory(folderPath.c_str(), NULL);
	if (flag == true) { cout << "文件夹" << folderPath << "创建成功" << endl; }
	else { cout << "Directory " << folderPath << " already exists." << endl; }
	return 0;
}

int main()
{

	Pylon::PylonAutoInitTerm autoInitTerm;
	//Pylon::PylonAutoInitTerm autoInitTerm2;
	try
	{

		CTlFactory& TlFactory = CTlFactory::GetInstance();
		//CTlFactory& TlFactory2 = CTlFactory::GetInstance();
		ITransportLayer* pTl = TlFactory.CreateTl(Camera_t::DeviceClass());
		//ITransportLayer* pT2 = TlFactory2.CreateTl(Camera_u::DeviceClass());






		DeviceInfoList_t devices;
		if (0 == pTl->EnumerateDevices(devices))
		{
			cerr << "No camera present!" << endl;
			//            return 2;
		}

		Camera_t Camera(pTl->CreateDevice(devices[0]));
		Camera_t Camera1(pTl->CreateDevice(devices[1]));
		CHeartbeatHelper headbeatHelper(Camera);
		CHeartbeatHelper headbeatHelper1(Camera1);
		headbeatHelper.SetValue(5000);
		headbeatHelper1.SetValue(5000);

		Camera.Open();
		Camera1.Open();
		Camera_t::StreamGrabber_t StreamGrabber(Camera.GetStreamGrabber(0));
		Camera_t::StreamGrabber_t StreamGrabber1(Camera1.GetStreamGrabber(0));
		StreamGrabber.Open();
		StreamGrabber1.Open();
		Camera.PixelFormat.SetValue(PixelFormat_Mono8);
		Camera1.PixelFormat.SetValue(PixelFormat_Mono8);
		//Camera.PixelFormat.SetValue(PixelFormat_BayerRG8);

		Camera.OffsetX.SetValue(OffsetX);
		Camera.OffsetY.SetValue(OffsetY);
		Camera.Width.SetValue(width);
		Camera.Height.SetValue(height);
		Camera1.OffsetX.SetValue(OffsetX);
		Camera1.OffsetY.SetValue(OffsetY);
		Camera1.Width.SetValue(width);
		Camera1.Height.SetValue(height);
		GenApi::IEnumEntry* frameStart = Camera.TriggerSelector.GetEntry(TriggerSelector_FrameStart);
		GenApi::IEnumEntry* frameStart1 = Camera1.TriggerSelector.GetEntry(TriggerSelector_FrameStart);

		Camera.AcquisitionMode.SetValue(AcquisitionMode_Continuous);
		Camera1.AcquisitionMode.SetValue(AcquisitionMode_Continuous);

		Camera.ExposureMode.SetValue(ExposureMode_Timed);
		//Camera.ExposureTimeAbs.SetValue(20);
		Camera.ExposureTimeRaw.SetValue(Exp);

		Camera.GainAuto.SetValue(GainAuto_Off);
		Camera.AcquisitionFrameRateEnable.SetValue(TRUE);
		Camera.AcquisitionFrameRateAbs.SetValue(1);
		Camera1.AcquisitionFrameRateEnable.SetValue(TRUE);
		Camera1.AcquisitionFrameRateAbs.SetValue(1);

		Camera1.ExposureMode.SetValue(ExposureMode_Timed);
		//Camera.ExposureTimeAbs.SetValue(20);
		Camera1.ExposureTimeRaw.SetValue(Exp);

		Camera1.GainAuto.SetValue(GainAuto_Off);

		const size_t ImageSize = (size_t)(Camera.PayloadSize.GetValue());
		const size_t ImageSize1 = (size_t)(Camera1.PayloadSize.GetValue());
		ushort* const pBuffer = new ushort[ImageSize];
		ushort* const pBuffer1 = new ushort[ImageSize1];

		StreamGrabber.MaxBufferSize.SetValue(ImageSize);
		StreamGrabber1.MaxBufferSize.SetValue(ImageSize1);
		StreamGrabber.MaxNumBuffer.SetValue(1);
		StreamGrabber1.MaxNumBuffer.SetValue(1);
		StreamGrabber.PrepareGrab();
		StreamGrabber1.PrepareGrab();
		const StreamBufferHandle hBuffer =
			StreamGrabber.RegisterBuffer(pBuffer, ImageSize);
		const StreamBufferHandle hBuffer1 =
			StreamGrabber1.RegisterBuffer(pBuffer1, ImageSize1);
		StreamGrabber.QueueBuffer(hBuffer, NULL);
		StreamGrabber1.QueueBuffer(hBuffer1, NULL);
		if (frameStart && GenApi::IsAvailable(frameStart))
		{
			Camera.TriggerSelector.SetValue(TriggerSelector_FrameStart);
			Camera.TriggerMode.SetValue(TriggerModeEnums(0));
		}
		if (frameStart1 && GenApi::IsAvailable(frameStart1))
		{
			Camera1.TriggerSelector.SetValue(TriggerSelector_FrameStart);
			Camera1.TriggerMode.SetValue(TriggerModeEnums(0));
		}
		cout << "Camera open!" << endl;


		time_t t1 = time(0);
		char tmp[64];
		strftime(tmp, sizeof(tmp), "%H_%M_%S", localtime(&t1));
		string s = tmp;
		savepath0 = savepath + s + "//";
		newfold(savepath0);

		savepath_1 = savepath0 + to_string(1) + "//";
		newfold(savepath_1);
		savepath_2 = savepath0 + to_string(2) + "//";
		newfold(savepath_2);
		int num = 0;
		while (1)
		{
			printf("按下任意键采集@_@");
			while (cin.get() == '\n')
			{
				break;
			}
			cout << num << endl;
			num++;
			Camera.AcquisitionStart.Execute();
			Camera1.AcquisitionStart.Execute();
			GrabResult Result;
			GrabResult Result1;
			if (StreamGrabber.GetWaitObject().Wait(3000))
			{

				//                                GrabResult Result;
				StreamGrabber.RetrieveResult(Result);

				memcpy(src.data, Result.Buffer(), src.cols * src.rows * sizeof(uchar));

				imwrite(savepath_1 + to_string(num) + ".jpg", src);


			}
			if (StreamGrabber1.GetWaitObject().Wait(3000))
			{
				//                                GrabResult Result;
				StreamGrabber1.RetrieveResult(Result1);

				memcpy(src1.data, Result1.Buffer(), src1.cols * src1.rows * sizeof(uchar));

				imwrite(savepath_2 + to_string(num) + ".jpg", src1);


			}
			StreamGrabber.QueueBuffer(Result.Handle(), NULL);
			StreamGrabber1.QueueBuffer(Result1.Handle(), NULL);
		}


		StreamGrabber.DeregisterBuffer(hBuffer);

		StreamGrabber.FinishGrab();

		StreamGrabber.Close();

		Camera.Close();
		delete[] pBuffer;
		StreamGrabber1.DeregisterBuffer(hBuffer1);

		StreamGrabber1.FinishGrab();

		StreamGrabber1.Close();

		Camera1.Close();
		delete[] pBuffer1;


	}

	catch (const GenericException& e)
	{
		// Error handling
	//        cerr << "An exception occurred." << endl
	//            << e.GetDescription() << endl;
	}


	//    return 0;
}
