#pragma once
#include "ISA.h"
#include "Utility.h"

#include <vector>
using namespace std;

// ProcessingDlg dialog

class ProcessingDlg : public CDialogEx
{
	DECLARE_DYNAMIC(ProcessingDlg)

public:
	ProcessingDlg(CImage * img, int width = 200, int height = 100, int imageType = 0, CWnd* pParent = nullptr);   // standard constructor
	virtual ~ProcessingDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_PROCESSING };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	DECLARE_MESSAGE_MAP()

protected:
	void PrintPicture(CImage *pImgSrc, CStatic &cPicPanel);
	void ClearPicture();
	void DrawPoint(CImage *pImg, CPoint &point, int r);
	void DrawPointList(CImage *pImg, vector<CPoint> &pointList, int r);

private:

	int m_ImgWidth;
	int m_ImgHeight;
	CRect m_ImgRect;

	CStatic m_ImgPanel;
	CImage * m_pImgSrc;
	vector<CImage*> m_SeedPointImgList;
	CImage * m_pImgTemp;
	ISA * m_Scissors;

	bool m_isDrawing;
	vector<CPoint> m_FirstSeedPointList;
	vector<unsigned int> m_FirstSeedPointIndexList;
	int m_nSeedPointNum;
public:
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg BOOL OnSetCursor(CWnd* pWnd, UINT nHitTest, UINT message);
	afx_msg void OnRButtonDown(UINT nFlags, CPoint point);
};
