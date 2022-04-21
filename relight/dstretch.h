#ifndef DSTRETCH_H
#define DSTRETCH_H

#include <imageset.h>
#include "jpeg_decoder.h"
#include "jpeg_encoder.h"
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <QString>
#include <QDebug>

inline void dstretchSingle(QString fileName, QString output, int minSamples, std::function<bool(std::string s, int n)> progressed)
{
    // Vector used to temporarily store a line of pixels
    std::vector<uint8_t> pixels;
    // Vector used to store samples
    std::vector<Color3b> samples;

    // Final data to be saved
    std::vector<int> dstretched;
    std::vector<uint8_t> dstretchedBytes;
    // Jpeg encoder and decoder to handle output and input
    JpegDecoder decoder;
    JpegEncoder encoder;

    // Size of the image
    int width, height;
    // Covariance matrix
    Eigen::MatrixXd covariance(3, 3);
    // Channel means
    Eigen::VectorXd means(3);

    // Max and min values for channels (used to rescale the output)
    int mins[] = {256, 256, 256};
    int maxs[] = {-1, -1, -1};

    // Initialization
    decoder.init(fileName.toStdString().c_str(), width, height);
    encoder.init(output.toStdString().c_str(), width, height);
    pixels.resize(width * 3);

    means.fill(0);

    // Compute the distances between a sample and another one so that at least minSamples are taken
    uint32_t samplesHorizontal = std::ceil(std::sqrt(minSamples) * ((float)width / height));
    uint32_t samplesVertical =  std::ceil(std::sqrt(minSamples) * ((float)height / width));
    uint32_t rowSkip = std::max<uint32_t>(1, height / samplesVertical);
    uint32_t colSkip = std::max<uint32_t>(1, width / samplesHorizontal);

    // Sample each line
    for (int row=0; row<height; row++)
    {
        // Read the line
        decoder.readRows(1, pixels.data());

        if (row % rowSkip == 0)
        {
            // Getting the samples
            for (int col=0; col<pixels.size(); col+=colSkip*3)
            {
                for (int i=0; i<3; i++)
                    means(i) += pixels[col + i];

                Color3b color(pixels[col], pixels[col+1], pixels[col+2]);
                samples.push_back(color);
            }
        }

        if (row % 10 == 0)
            progressed("Sampling...", (row * 100) / width);
    }

    // Compute the mean of the channels
    for (int i=0; i<3; i++)
        means(i) /= samples.size();

    // Compute the sums needed to compute the covariance
    long sumChannel[]= {0,0,0};
    double sumX[][3] = {{0,0,0},{0,0,0},{0,0,0}};

    for (int k=0; k<samples.size(); k++)
        for (int i=0; i<3; i++)
            sumChannel[i] += samples[k][i];

    for (int l=0; l<3; l++)
        for (int m=0; m<3; m++)
            for (int k=0; k<samples.size(); k++)
                sumX[l][m] += samples[k][l] * samples[k][m];

    // Compute the covariance
    for (int l=0; l<3; l++)
        for (int m=0; m<3; m++)
            covariance(l,m) = ((double)(1.0f/(samples.size() - 1))) * (sumX[l][m] - ((double)1.0f/samples.size())*sumChannel[l]*sumChannel[m]);

    // Compute the rotation
    Eigen::EigenSolver<Eigen::MatrixXd> solver(covariance, true);
    Eigen::MatrixXd rotation = solver.eigenvectors().real();
    Eigen::MatrixXd eigenValues = solver.eigenvalues().real();

    Eigen::MatrixXd sigma = covariance.diagonal().asDiagonal();
    for (int i=0; i<3; i++)
        sigma(i, i) = std::sqrt(sigma(i,i));

    // Compute the stretching factor
    for (int i=0; i<3; i++)
        eigenValues(i) = 1.0f / std::sqrt(eigenValues(i) >= 0 ? eigenValues(i) : -eigenValues(i));

    // Compute the final transformation matrix
    Eigen::MatrixXd transformation = sigma * rotation * eigenValues.asDiagonal() * rotation.transpose();
    // Apply the transformation to the mean
    Eigen::VectorXd offset = means - transformation * means;

    // Finally reposition the pixels with that offset
    decoder.init(fileName.toStdString().c_str(), width, height);
    Eigen::VectorXd currPixel(3);

    for (int i=0; i<height; i++)
    {
        decoder.readRows(1, pixels.data());
        for (int k=0; k<pixels.size(); k+=3)
        {
            for (int j=0; j<3; j++)
                currPixel(j) = pixels[k+j];

            currPixel -= means;
            currPixel = transformation * currPixel + means + offset;

            for (int j=0; j<3; j++)
            {
                dstretched.push_back(currPixel[j]);
                mins[j] = std::min<int>(mins[j], currPixel[j]);
                maxs[j] = std::max<int>(maxs[j], currPixel[j]);
            }
        }

        if (i % 100 == 0)
            progressed("Transforming...", (i * 100) / height);
    }

    for (int k=0; k<dstretched.size(); k++)
    {
        uint32_t channelIdx = k % 3;
        dstretchedBytes.push_back(255 * ((float)(dstretched[k] - mins[channelIdx]) / (maxs[channelIdx] - mins[channelIdx])));

        if (k % 100 == 0)
            progressed("Scaling...", (k * 100) / dstretched.size());
    }

    progressed("Saving...", 50);
    encoder.writeRows(dstretchedBytes.data(), height);
    encoder.finish();
}


inline void dstretchSet(QString inputFolder, QString output, int minSamples, std::function<bool(std::string s, int n)> progressed)
{
    output = output.mid(0, output.lastIndexOf("/"));
    qDebug() << "SETT";
    // Image set initialization
    ImageSet set;
    set.setCallback(nullptr);
    set.initFromFolder(inputFolder.toStdString().c_str());
    int nImages = set.images.size();

    // Vector used to temporarily store a line of pixels
    PixelArray pixels;
    // Vector used to store samples
    std::vector<std::vector<Color3f>> samples;

    // Final data to be saved
    std::vector<std::vector<float>> dstretched;
    std::vector<std::vector<uint8_t>> dstretchedBytes;

    // Size of the image
    int width, height;
    // Covariance matrix
    Eigen::MatrixXd covariance(3 * nImages, 3 * nImages);
    // Channel means
    Eigen::VectorXd means(nImages * 3);

    // Max and min values for channels (used to rescale the output)
    float** mins, **maxs;

    mins = new float*[nImages];
    maxs = new float*[nImages];
    for (int i=0; i<nImages; i++)
    {
        mins[i] = new float[3];
        maxs[i] = new float[3];

        for (int j=0; j<3; j++)
        {
            mins[i][j] = 2048;
            maxs[i][j] = -2048;
        }
    }


    width = set.width;
    height = set.height;
    pixels.resize(width, nImages);
    samples.resize(nImages);
    means.fill(0);

    dstretched.resize(nImages);
    dstretchedBytes.resize(nImages);

    // Compute the distances between a sample and another one so that at least minSamples are taken
    uint32_t samplesHorizontal = std::ceil(std::sqrt(minSamples) * ((float)width / height));
    uint32_t samplesVertical =  std::ceil(std::sqrt(minSamples) * ((float)height / width));
    uint32_t rowSkip = std::max<uint32_t>(1, height / samplesVertical);
    uint32_t colSkip = std::max<uint32_t>(1, width / samplesHorizontal);

    // Sample each line
    for (int row=0; row<height; row++)
    {
        // Read the line
        set.readLine(pixels);

        if (row % rowSkip == 0)
        {
            // Getting the samples
            for (int col=0; col<pixels.size(); col+=colSkip)
            {
                for (int l=0; l<nImages; l++)
                    for (int i=0; i<3; i++)
                        // For each light we have a line, take the col pixel from that line and add the i channel
                        means(l*3 + i) += pixels[col][l][i];

                for (int l=0; l<nImages; l++)
                    samples[l].push_back(pixels[col][l]);
            }
        }

        if (row % 10 == 0)
            progressed("Sampling...", (row * 100) / width);
    }

    // Compute the mean of the channels
    for (int im=0; im<samples.size(); im++)
        for (int i=0; i<3; i++)
            means(im*3 + i) /= samples[im].size();

    // Compute the sums needed to compute the covariance
    // Initialize the matrices
    std::vector<long[3]> sumChannel(nImages);
    std::vector<std::vector<double>> sumX(nImages*3);
    for (int i=0; i<nImages*3; i++)
        sumX[i].resize(nImages*3);

    // Compute the channel sums
    for (int im=0; im<nImages; im++)
        for (int k=0; k<samples[im].size(); k++)
            for (int i=0; i<3; i++)
                sumChannel[im][i] += samples[im][k][i];

    // Compute the covariance sums
    for (int l=0; l<nImages*3; l++)
    {
        for (int m=0; m<nImages*3; m++)
        {
            int imgL = l / 3;
            int imgM = m / 3;
            int lChannel = l % 3;
            int mChannel = m % 3;

            for (int k=0; k<samples[imgL].size(); k++)
                sumX[l][m] += samples[imgL][k][lChannel] * samples[imgM][k][mChannel];
        }
    }

    // Compute the covariance
    for (int l=0; l<nImages*3; l++)
    {
        for (int m=0; m<nImages*3; m++)
        {
            int imgL = l / 3;
            int imgM = m / 3;

            covariance(l,m) =   ((double)(1.0f/(samples[imgL].size() - 1))) *
                                (sumX[l][m] - ((double)1.0f/samples[imgL].size())*sumChannel[imgL][l%3]*sumChannel[imgM][m%3]);

            double sas = covariance(l,m);
            qDebug() << sas;
        }
    }

    // Compute the rotation
    Eigen::EigenSolver<Eigen::MatrixXd> solver(covariance, true);
    Eigen::MatrixXd rotation = solver.eigenvectors().real();
    Eigen::MatrixXd eigenValues = solver.eigenvalues().real();

    Eigen::MatrixXd sigma = covariance.diagonal().asDiagonal();
    for (int i=0; i<nImages*3; i++)
        sigma(i, i) = std::sqrt(sigma(i,i));

    // Compute the stretching factor
    for (int i=0; i<nImages*3; i++)
        eigenValues(i) = 1.0f / std::sqrt(eigenValues(i) >= 0 ? eigenValues(i) : -eigenValues(i));

    // Compute the final transformation matrix
    Eigen::MatrixXd transformation = sigma * rotation * eigenValues.asDiagonal() * rotation.transpose();
    // Apply the transformation to the mean
    Eigen::VectorXd offset = means - transformation * means;

    // Start up the encoders
    JpegEncoder* encoders = new JpegEncoder[nImages];

    // Transform and scale all the images in different files
    Eigen::VectorXd currPixel(3 * nImages);
    PixelArray line;
    set.restart();

    for (int i=0; i<height; i++)
    {
        // Read a line
        set.readLine(line);

        for (int k=0; k<line.size(); k++)
        {
            // Create the pixel vector, that contains all the nImages pixels of the set
            for (int l=0; l<line[k].size(); l++)
                for (int c=0; c<3; c++)
                    currPixel(l*3 + c) = line[k][l][c];

            // Transform the pixel
            currPixel -= means;
            currPixel = transformation * currPixel + means + offset;

            // Save the data for later scaling
            for (int l=0; l<currPixel.size(); l+=3)
            {
                for (int c=0; c<3; c++)
                {
                    dstretched[l/3].push_back(currPixel(l + c));
                    mins[l/3][c] = std::min<int>(mins[l/3][c], currPixel(l + c));
                    maxs[l/3][c] = std::max<int>(maxs[l/3][c], currPixel(l + c));
                }
            }
        }

        if (i % 100 == 0)
            progressed("Transforming...", (i * 100) / height);
    }

    qDebug() << "Out files: " << QString(output + "/img_%1.jpg").arg(1).toStdString().c_str();

    for (int im=0; im<nImages; im++)
    {
        for (int k=0; k<dstretched[im].size(); k++)
        {
            uint32_t channelIdx = k % 3;
            dstretchedBytes[im].push_back(255 * ((float)(dstretched[im][k] - mins[im][channelIdx]) / (maxs[im][channelIdx] - mins[im][channelIdx])));
        }

        encoders[im].encode(dstretchedBytes[im].data(), width, height, (inputFolder + "/" + set.images[im]).toStdString().c_str());
        progressed("Scaling...", (im * 100) / nImages);
    }
}

inline void dstretch()
{

}


#endif // DSTRETCH_H

