#include <iostream>
#include <cmath>
#include <time.h>

// VTK includes.
#include <vtkDataSetReader.h>
#include <vtkSmartPointer.h>
#include <vtkRectilinearGrid.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkPNGWriter.h>
#include <vtkImageData.h>


const double PI = 3.141592653589793238463;  // Approximate pi constant.


using namespace std;

struct Camera
{
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];
};


struct TransferFunction
{
    double          min;
    double          max;
    int             numBins;
    unsigned char  *colors;  // size is 3*numBins
    double         *opacities; // size is numBins

    // Take in a value and applies the transfer function.
    // Step #1: figure out which bin "value" lies in.
    // If "min" is 2 and "max" is 4, and there are 10 bins, then
    //   bin 0 = 2->2.2
    //   bin 1 = 2.2->2.4
    //   bin 2 = 2.4->2.6
    //   bin 3 = 2.6->2.8
    //   bin 4 = 2.8->3.0
    //   bin 5 = 3.0->3.2
    //   bin 6 = 3.2->3.4
    //   bin 7 = 3.4->3.6
    //   bin 8 = 3.6->3.8
    //   bin 9 = 3.8->4.0
    // and, for example, a "value" of 3.15 would return the color in bin 5
    // and the opacity at "opacities[5]".
    void ApplyTransferFunction(double value, unsigned char *RGB, double &opacity)
    {
        int bin = GetBin(value);
        RGB[0] = colors[3*bin+0];
        RGB[1] = colors[3*bin+1];
        RGB[2] = colors[3*bin+2];
        opacity = opacities[bin];
    }

private:
    int GetBin(double value)
    {
        double binSize = (max - min) / numBins;
        int i = 0;
        for (; i < numBins; i++)
        {
            value -= binSize;
            if (value < min)
            {
                break;
            }
        }
        return i;
    }
};

TransferFunction
SetupTransferFunction(void)
{
    int  i;

    TransferFunction rv;
    rv.min = 10;
    rv.max = 15;
    rv.numBins = 256;
    rv.colors = new unsigned char[3*256];
    rv.opacities = new double[256];
    unsigned char charOpacity[256] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14, 14, 14, 14, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 5, 4, 3, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 16, 16, 15, 14, 13, 12, 11, 9, 8, 7, 6, 5, 5, 4, 3, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 18, 20, 22, 24, 27, 29, 32, 35, 38, 41, 44, 47, 50, 52, 55, 58, 60, 62, 64, 66, 67, 68, 69, 70, 70, 70, 69, 68, 67, 66, 64, 62, 60, 58, 55, 52, 50, 47, 44, 41, 38, 35, 32, 29, 27, 24, 22, 20, 20, 23, 28, 33, 38, 45, 51, 59, 67, 76, 85, 95, 105, 116, 127, 138, 149, 160, 170, 180, 189, 198, 205, 212, 217, 221, 223, 224, 224, 222, 219, 214, 208, 201, 193, 184, 174, 164, 153, 142, 131, 120, 109, 99, 89, 79, 70, 62, 54, 47, 40, 35, 30, 25, 21, 17, 14, 12, 10, 8, 6, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

    for (i = 0 ; i < 256 ; i++)
        rv.opacities[i] = charOpacity[i]/255.0;
    const int numControlPoints = 8;
    unsigned char controlPointColors[numControlPoints*3] = { 
           71, 71, 219, 0, 0, 91, 0, 255, 255, 0, 127, 0, 
           255, 255, 0, 255, 96, 0, 107, 0, 0, 224, 76, 76 
       };
    double controlPointPositions[numControlPoints] = { 0, 0.143, 0.285, 0.429, 0.571, 0.714, 0.857, 1.0 };
    for (i = 0 ; i < numControlPoints-1 ; i++)
    {
        int start = controlPointPositions[i]*rv.numBins;
        int end   = controlPointPositions[i+1]*rv.numBins+1;
        cerr << "Working on " << i << "/" << i+1 << ", with range " << start << "/" << end << endl;
        if (end >= rv.numBins)
            end = rv.numBins-1;
        for (int j = start ; j <= end ; j++)
        {
            double proportion = (j/(rv.numBins-1.0)-controlPointPositions[i])/(controlPointPositions[i+1]-controlPointPositions[i]);
            if (proportion < 0 || proportion > 1.)
                continue;
            for (int k = 0 ; k < 3 ; k++)
                rv.colors[3*j+k] = proportion*(controlPointColors[3*(i+1)+k]-controlPointColors[3*i+k])
                                 + controlPointColors[3*i+k];
        }
    }    

    return rv;
}

Camera
SetupCamera(void)
{
    Camera rv;
    rv.focus[0] = 0;
    rv.focus[1] = 0;
    rv.focus[2] = 0;
    rv.up[0] = 0;
    rv.up[1] = -1;
    rv.up[2] = 0;
    rv.angle = 30;
    rv.near = 7.5e+7;
    rv.far = 1.4e+8;
    rv.position[0] = -8.25e+7;
    rv.position[1] = -3.45e+7;
    rv.position[2] = 3.35e+7;

    return rv;
}


void ChangeInPosition(const double angle, const double length, const double* v, double* res)
{
    // Convert the angle in degrees to radians.
    double angleRadians = (PI / 180) * angle;

    res[0] = (2 * tan(angleRadians / 2) * v[0]) / length;
    res[1] = (2 * tan(angleRadians / 2) * v[1]) / length;
    res[2] = (2 * tan(angleRadians / 2) * v[2]) / length;
}


void NormalizeCrossProduct(const double* v1, const double* v2, double* ans)
{
    double top[3];
    double bot[3];
    top[0] = (v1[1] * v2[2]) - (v2[1] * v1[2]);
    top[1] = -1 * ((v1[0] * v2[2]) - (v2[0] * v1[2]));
    top[2] = (v1[0] * v2[1]) - (v2[0] * v1[1]);

    double normalized_cross_prod = sqrt(top[0] * top[0] + top[1] * top[1] + top[2] * top[2]);
    
    ans[0] = top[0] / normalized_cross_prod;
    ans[1] = top[1] / normalized_cross_prod;
    ans[2] = top[2] / normalized_cross_prod;
}


void CalculateRay(const Camera cam, const double xSize, const double ySize, const int i, const int j, double* ray)
{
    double R_u[3];
    double R_v[3];
    double R_x[3];
    double R_y[3];

    double look[3] = {
        cam.focus[0] - cam.position[0],
        cam.focus[1] - cam.position[1],
        cam.focus[2] - cam.position[2]
    };

    NormalizeCrossProduct(look, cam.up, R_u);
    NormalizeCrossProduct(look, R_u, R_v);

    ChangeInPosition(cam.angle, xSize, R_u, R_x);
    ChangeInPosition(cam.angle, ySize, R_v, R_y);

    double look_mag = sqrt(look[0] * look[0] + look[1] * look[1] + look[2] * look[2]);
    ray[0] = (look[0] / look_mag) + ((2.0 * i + 1.0 - xSize) / 2.0) * R_x[0] +
        ((2 * j + 1 - ySize) / 2.0) * R_y[0];
    ray[1] = (look[1] / look_mag) + ((2.0 * i + 1.0 - xSize) / 2.0) * R_x[1] +
        ((2 * j + 1 - ySize) / 2.0) * R_y[1];
    ray[2] = (look[2] / look_mag) + ((2.0 * i + 1.0 - xSize) / 2.0) * R_x[2] +
        ((2 * j + 1 - ySize) / 2.0) * R_y[2];
}


int FindIndex(float* Arr, int length, double value)
{
    for (int k = 0; k < length; k++)
    {
        if (value < Arr[k])
        {
            return k;
        }
    }
}


double TrilinearInterpolation(const int* idx, const int* dims, const double* pos, const float* X, const float* Y, const float* Z, const float* F)
{
    // Interpolate along the top front of the cell.
    float pt0 = X[idx[0] - 1];
    float t = (pos[0] - pt0) / (X[idx[0]] - pt0);
    float r_c_v = F[idx[2] * dims[0] * dims[1] + idx[1] * dims[0] + idx[0]];
    float l_c_v = F[idx[2] * dims[0] * dims[1] + idx[1] * dims[0] + (idx[0] - 1)];
    float front = l_c_v + t * (r_c_v - l_c_v);
    
    // Interpolate along the back front of the cell.
    pt0 = X[idx[0] - 1];
    t = (pos[0] - pt0) / (X[idx[0]] - pt0);
    r_c_v = F[(idx[2] - 1) * dims[0] * dims[1] + idx[1] * dims[0] + idx[0]];
    l_c_v = F[(idx[2] - 1) * dims[0] * dims[1] + idx[1] * dims[0] + (idx[0] - 1)];
    float back = l_c_v + t * (r_c_v - l_c_v);

    // Interpolate between.
    pt0 = Z[idx[2] - 1];
    t = (pos[2] - pt0) / (Z[idx[2]] - pt0);
    float top_face_val = back + t * (front - back);
    

    // Interpolate along the bottom front of the cell.
    pt0 = X[idx[0] - 1];
    t = (pos[0] - pt0) / (X[idx[0]] - pt0);
    r_c_v = F[idx[2] * dims[0] * dims[1] + (idx[1] - 1) * dims[0] + idx[0]];
    l_c_v = F[idx[2] * dims[0] * dims[1] + (idx[1] - 1) * dims[0] + (idx[0] - 1)];
    front = l_c_v + t * (r_c_v - l_c_v);

    // Interpolate along the bottom back of the cell.
    pt0 = X[idx[0] - 1];
    t = (pos[0] - pt0) / (X[idx[0]] - pt0);
    r_c_v = F[(idx[2] - 1) * dims[0] * dims[1] + (idx[1] - 1) * dims[0] + idx[0]];
    l_c_v = F[(idx[2] - 1) * dims[0] * dims[1] + (idx[1] - 1) * dims[0] + (idx[0] - 1)];
    back = l_c_v + t * (r_c_v - l_c_v);

    // Interpolate between.
    pt0 = Z[idx[2] - 1];
    t = (pos[2] - pt0) / (Z[idx[2]] - pt0);
    float bot_face_val = back + t * (front - back);


    // Interopolate bewtewen the top and bottom faces.
    pt0 = Y[idx[1] - 1];
    t = (pos[1] - pt0) / (Y[idx[1]] - pt0);
    
    return bot_face_val + t * (top_face_val - bot_face_val);
}


void WriteImage(vtkImageData* image)
{
    vtkNew<vtkPNGWriter> writer;
    writer->SetInputData(image);
    writer->SetFileName("output.png");
    writer->Write();
}


typedef struct TimeValues
{
    clock_t read_time;
    clock_t program_start;
    clock_t begin_processing;
} ProgramTimes;


int main(int argc, char** argv)
{
    ProgramTimes pt;
    clock_t clock_start;
    pt.program_start = clock();

    //const int samplesPerRay = 256;
    const int samplesPerRay = 1024; 

    //int xSize = 100;
    //int ySize = 100;
    int xSize = 1000;
    int ySize = 1000;

    // Create the data reader
    vtkNew<vtkDataSetReader> reader;

    cerr << "Starting Volume Render... " << endl;
    
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <input-file> <samples-per-ray>" << endl;
        exit(EXIT_FAILURE);
    }
    
    std::string inputFile = argv[1];
    reader->SetFileName(inputFile.c_str());

    cerr << "Reading input file... " << endl;
    
    clock_start = clock();
    reader->Update();
    pt.read_time = float(clock() - clock_start) / CLOCKS_PER_SEC;

    if (reader->GetOutput() == NULL || reader->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Could not find input file." << endl;
        exit(EXIT_FAILURE);
    }

    
    vtkRectilinearGrid* rgrid = (vtkRectilinearGrid*)reader->GetOutput();
    int dims[3];
    rgrid->GetDimensions(dims);
    float* X = (float*)rgrid->GetXCoordinates()->GetVoidPointer(0);
    float* Y = (float*)rgrid->GetYCoordinates()->GetVoidPointer(0);
    float* Z = (float*)rgrid->GetZCoordinates()->GetVoidPointer(0);
    float* F = (float*)rgrid->GetPointData()->GetScalars()->GetVoidPointer(0); 
    int ncells = rgrid->GetNumberOfCells();

    // Create the image and assign a buffer to write pixel data for the image.
    vtkNew<vtkImageData> image;
    image->SetDimensions(xSize, ySize, 1);
    image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    
    unsigned char* buffer;
    buffer = (unsigned char*)image->GetScalarPointer(0, 0, 0);
    for (int i = 0; i < 3 * xSize * ySize; i++)
    {
        buffer[i] = 0;
    }

    // Set up the camera and transfer function.
    TransferFunction tf = SetupTransferFunction();
    Camera cam = SetupCamera();

    // Alloacate a buffer for the samples and associated values to store position data scalar values for each sample 
    // along the ray cast.
    double** sample = new double* [samplesPerRay];
    double* value = new double[samplesPerRay];
    for (int i = 0; i < samplesPerRay; i++)
    {
        sample[i] = new double[3];
    }

    
    cerr << "Looping through pixels..." << endl;


    pt.begin_processing = clock();
    // Iterate over every pixel. 
    for (int i = 0; i < xSize; i++)
    {
        for (int j = 0; j < ySize; j++)
        {
            // Calculate the ray.
            double ray[3];
            CalculateRay(cam, xSize, ySize, i, j, ray);


            // The first sample is right after the near frame of the camera, so samplePerRay - 1 is needed to divide the 
            // other points along the ray evenly.
            // Double used because we lose some accuracy when converting into an integer.
            double stepSize = (cam.far - cam.near) / (samplesPerRay - 1);

            double pos[] = { cam.position[0] + cam.near * ray[0], (cam.position[1] + cam.near * ray[1]) , (cam.position[2] + cam.near * ray[2]) };
            int idx[3] = { 0, 0, 0 };

            for (int s = 0; s < samplesPerRay; s++)
            {
                sample[s][0] = pos[0];
                sample[s][1] = pos[1];
                sample[s][2] = pos[2];

                
                idx[0] = FindIndex(X, dims[0], pos[0]);  
                idx[1] = FindIndex(Y, dims[1], pos[1]);
                idx[2] = FindIndex(Z, dims[2], pos[2]);

                if (idx[0] <= 0 || idx[1] <= 0 || idx[2] <= 0)
                    continue;

                // Calculate the scalar value at a cell with indices of idx.
                value[s] = TrilinearInterpolation(idx, dims, pos, X, Y, Z, F); 

                pos[0] = pos[0] + ray[0] * stepSize;
                pos[1] = pos[1] + ray[1] * stepSize;
                pos[2] = pos[2] + ray[2] * stepSize;
            }


            // Initialize the running totals for the color. 
            double runningRGB[] = { 0, 0, 0 };
            double runningOpacity = 0.0;

            for (int s = 0; s < samplesPerRay; s++)
            {
                unsigned char RGB[] = { 0, 0, 0 };
                double opacity = 0.0;
                double correctedOp = 0.0;
                
                // Check to see if the value is within the transfer function range.
                if (tf.min < value[s] && value[s] < tf.max)
                {
                    tf.ApplyTransferFunction(value[s], RGB, opacity);

                    // Correct the opacity using a default of 500 samples per ray.
                    correctedOp = 1 - pow(1 - opacity, 500.0 / samplesPerRay);

                    runningRGB[0] = runningRGB[0] + (1 - runningOpacity) * (RGB[0] / 255.0) * correctedOp;

                    runningRGB[1] = runningRGB[1] + (1 - runningOpacity) * (RGB[1] / 255.0) * correctedOp;

                    runningRGB[2] = runningRGB[2] + (1 - runningOpacity) * (RGB[2] / 255.0) * correctedOp;
                    runningOpacity = runningOpacity + (1 - runningOpacity) * correctedOp;

                    if (runningOpacity > 0.95)
                    {
                        break;
                    }
                }
            }

            // Write the RGB values to the buffer.
            buffer[3 * (j * xSize + i)] = runningRGB[0] * 255;
            buffer[3 * (j * xSize + i) + 1] = runningRGB[1] * 255;
            buffer[3 * (j * xSize + i) + 2] = runningRGB[2] * 255;
        }
        //cerr << "Row " << i << " complete." << endl;
    }

    cerr << "File read time: " << pt.read_time << endl;

    cerr << "Processing time: " << float(clock() - pt.begin_processing) / CLOCKS_PER_SEC << endl;
      
    WriteImage(image);
    for (int i = 0; i < samplesPerRay; i++)
    {
        delete sample[i];
    }
    
    delete[] sample;
    delete[] value;
    cerr << "Total time: " << float(clock() - pt.program_start) / CLOCKS_PER_SEC << endl;

    cerr << "Volume render of input file saved as output.txt in same the directory " << endl;


    exit(EXIT_SUCCESS);
}