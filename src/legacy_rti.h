#ifndef RRTI_H
#define RRTI_H

#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <string>

/* NOTE: the image is flipped in the coefficients! */

class LRti {
public:
	enum Type { UNKNOWN = 0, PTM_LRGB = 1, PTM_RGB = 2, HSH  = 3};
	enum PTMFormat { RAW = 0, JPEG = 1, JPEGLS = 2 };

	Type type;

	int width;
	int height;
	std::vector<std::vector<unsigned char>> data; //organized as planes where for PTM is r g b ... r g b (18 planes)
	std::vector<float> scale;
	std::vector<float> bias;
	bool chromasubsampled = false;
	std::string error;

	LRti():  type(UNKNOWN), width(0), height(0) {}
	bool load(const char *filename);

	void clip(int left, int bottom, int right, int top);
	LRti clipped(int left, int bottom, int right, int top);

	bool encode(PTMFormat format, int &size, uint8_t *&buffer, int quality = 90);
	bool encode(PTMFormat format, const char *filename, int quality = 90);

	bool encodeJPEGtoFile(int startplane, int quality, const char *filename);
	//used to load relight planes into this class.
	bool decodeJPEGfromFile(size_t size, unsigned char *buffer, uint32_t plane0, uint32_t plane1, uint32_t plane2);


protected:
	bool loadPTM(FILE *file);
	bool loadHSH(FILE *file);

	bool decodeRAW(const std::string &version, FILE *file);
	bool decodeJPEG(FILE *file);

	//load jpeg into data;
	bool decodeJPEG(size_t size, unsigned char *buffer, uint32_t plane);
	bool encodeJPEG(std::vector<int> &sizes, std::vector<uint8_t *> &buffers, int quality = 90);
};

#endif // RRTI_H

