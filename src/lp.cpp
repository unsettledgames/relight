#include "lp.h"

#include <QFile>
#include <QTextStream>

using namespace std;

void parseLP(QString sphere_path, std::vector<Vector3f> &lights, std::vector<QString> &filenames) {
	QFile sphere(sphere_path);
	if(!sphere.open(QFile::ReadOnly))
		throw QString("Could not open: " + sphere_path);

	QTextStream stream(&sphere);
	bool ok;
	int n = stream.readLine().toInt(&ok);

	if(!ok || n <= 0 || n > 1000)
		throw QString("Invalid format or number of lights in .lp.");

	for(int i = 0; i < n; i++) {
		QString filename;
		Vector3f light;
		QString line = stream.readLine();
		QStringList tokens = line.split(QRegExp("\\s+"), QString::SkipEmptyParts);
		if(tokens.size() != 4)
			throw QString("Invalid line in .lp: " + line);

		filename = tokens[0];
		for(int k = 0; k < 3; k++) {
			bool ok;
			light[k] = tokens[k+1].toDouble(&ok);
			if(!ok)
				throw QString("Failed reading light direction in: " + line);
		}
		double norm = light.norm();
		if(norm < 0.0001)
			throw QString("Light direction too close to the origin! " + line);

		lights.push_back(light);
		filenames.push_back(filename);
	}
}
