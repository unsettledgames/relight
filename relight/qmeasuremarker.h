#ifndef QMarker_H
#define QMarker_H

#include "qmarker.h"



class Measure;
class QGraphicsPathItem;
class QGraphicsLineItem;
class QGraphicsTextItem;

class QMeasureMarker: public QMarker {
	Q_OBJECT
public:
	Measure *measure = nullptr;

	explicit QMeasureMarker(Measure *m, QGraphicsView *_view, QWidget *parent = nullptr);
	~QMeasureMarker();

	virtual void click(QPointF pos);
	virtual void cancelEditing();

	virtual void setSelected(bool value = true);
	void startMeasure();
	void askMeasure();

public slots:
	virtual void onEdit();

protected:
	//waiting for first point, waiting for second point, all done.

	enum Measuring { FIRST_POINT = 0, SECOND_POINT = 1, DONE = 2 };
	Measuring measuring = FIRST_POINT;

	QGraphicsPathItem *first = nullptr;
	QGraphicsPathItem *second = nullptr;
	QGraphicsLineItem *line = nullptr;
	QGraphicsTextItem *text = nullptr;
};



#endif // QMarker_H