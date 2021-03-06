// ProcessingDlg.cpp : implementation file
//

#include "stdafx.h"
#include "IntelligentScissors.h"
#include "ProcessingDlg.h"
#include "afxdialogex.h"

#include <conio.h>

#define IMAGE_PANEL_X 0
#define IMAGE_PANEL_Y 0
#define IMAGE_PANEL_BORDER 15
#define DIALOG_X 0
#define DIALOG_Y 0
#define DIALOG_EXTRA_WIDTH (2 * IMAGE_PANEL_X + 15)
#define DIALOG_EXTRA_HEIGHT (IMAGE_PANEL_X + IMAGE_PANEL_Y + 38)


// ProcessingDlg dialog

IMPLEMENT_DYNAMIC(ProcessingDlg, CDialogEx)

ProcessingDlg::ProcessingDlg(CImage * img, int width, int height, int imageType, CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_PROCESSING, pParent)
	, m_pImgSrc(img)
	, m_ImgWidth(width + IMAGE_PANEL_BORDER)
	, m_ImgHeight(height + IMAGE_PANEL_BORDER)
	, m_isDrawing(false)
	, m_nSeedPointNum(0)
{
	_cprintf("Preprocessing image... \n");
	m_Scissors = new ISA(img, imageType);
	_cprintf("Preprocessing succeed. \n");
	if(imageType == 0)
		m_SeedPointImgList.push_back(Utility::CloneImage(img));
	else
		m_SeedPointImgList.push_back(m_Scissors->getPreprocessedImg());
}

ProcessingDlg::~ProcessingDlg()
{
	delete m_Scissors;

	for(auto &img : m_SeedPointImgList)
	{
		if(img != nullptr)
			img->Destroy();
	}
	_cprintf("\nEnd process image.\n\n");
}

void ProcessingDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_STATIC_IMAGEPROCESSING, m_ImgPanel);
}

BOOL ProcessingDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	MoveWindow(DIALOG_X, DIALOG_Y, m_ImgWidth + DIALOG_EXTRA_WIDTH, m_ImgHeight + DIALOG_EXTRA_HEIGHT);
	m_ImgPanel.MoveWindow(IMAGE_PANEL_X, IMAGE_PANEL_Y, m_ImgWidth, m_ImgHeight);

	int height = m_pImgSrc->GetHeight();
	int width = m_pImgSrc->GetWidth();
	CRect panel_rect;
	m_ImgPanel.GetClientRect(&panel_rect);

	m_ImgRect = { 
		CPoint((int) ((panel_rect.Width() - width) / 2), (int) ((panel_rect.Height() - height) / 2)),
		CSize((int) width, (int) height)
	};

	return TRUE;
}

void ProcessingDlg::OnPaint()
{
	PrintPicture(m_SeedPointImgList.back(), m_ImgPanel);

	CDialogEx::OnPaint();
}

void ProcessingDlg::PrintPicture(CImage * pImgSrc, CStatic & cPicPanel)
{
	if(pImgSrc != NULL)
	{
		CDC *pDC = cPicPanel.GetDC();
		if(pDC != nullptr)
		{
			SetStretchBltMode(pDC->m_hDC, STRETCH_HALFTONE);
			pImgSrc->StretchBlt(pDC->m_hDC, m_ImgRect, SRCCOPY);
			ReleaseDC(pDC);
		}
	}
}

void ProcessingDlg::ClearPicture()
{
	CRect rect;
	m_ImgPanel.GetClientRect(&rect);
	m_ImgPanel.GetDC()->FillSolidRect(rect.left + 1, rect.top + 1, rect.Width() - 2, rect.Height() - 2, RGB(240, 240, 240));
}

void ProcessingDlg::DrawPoint(CImage * pImg, CPoint & point, int r)
{
	HDC hDC = pImg->GetDC();
	CDC *pDC = new CDC;
	pDC->Attach(hDC);

	CBrush brush(RGB(255, 0, 0)), *oldbrush;
	oldbrush = pDC->SelectObject(&brush);
	pDC->SelectStockObject(NULL_PEN);

	pDC->Ellipse(
		point.x - r,
		point.y - r,
		point.x + r,
		point.y + r
	);

	pDC->SelectObject(oldbrush);
	ReleaseDC(pDC);
	pImg->ReleaseDC();
}

void ProcessingDlg::DrawPointList(CImage * pImg, vector<CPoint>& pointList, int r)
{
	HDC hDC = pImg->GetDC();
	CDC *pDC = new CDC;
	pDC->Attach(hDC);

	CBrush brush(RGB(255, 0, 0)), *oldbrush;
	oldbrush = pDC->SelectObject(&brush);
	pDC->SelectStockObject(NULL_PEN);

	for(auto p : pointList)
	{
		pDC->Ellipse(p.x - r, p.y - r, p.x + r, p.y + r);
	}

	pDC->SelectObject(oldbrush);
	ReleaseDC(pDC);
	pImg->ReleaseDC();
}

BEGIN_MESSAGE_MAP(ProcessingDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_LBUTTONDOWN()
	ON_WM_MOUSEMOVE()
	ON_WM_SETCURSOR()
	ON_WM_RBUTTONDOWN()
END_MESSAGE_MAP()


// ProcessingDlg message handlers


void ProcessingDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	static int point_r = 4;

	// TODO: Add your message handler code here and/or call default
	if(point.x < m_ImgRect.left || point.x > m_ImgRect.right || point.y < m_ImgRect.top || point.y > m_ImgRect.bottom)
		return;

	CPoint seedpoint = { point.x - m_ImgRect.left, point.y - m_ImgRect.top };

	// when line is closed, stop drawing
	if(m_nSeedPointNum != 0 && seedpoint == m_FirstSeedPointList.back())
	{
		_cprintf("\nClosed curve is created.\n");
		m_SeedPointImgList.push_back(Utility::CloneImage(m_pImgTemp));
		m_isDrawing = false;
		return;
	}

	_cprintf("\nSet seed point at (%d, %d), wait to creat shortest path graph...\n", seedpoint.x, seedpoint.y);
	m_Scissors->setSeedPoint(seedpoint);
	_cprintf("Set seed point succeed.\n");

	m_nSeedPointNum++;

	// clone an image
	if(!m_isDrawing)
	{
		// if the seed point is the first one, record it and clone the last image
		m_SeedPointImgList.push_back(Utility::CloneImage(m_SeedPointImgList.back()));
		m_FirstSeedPointList.push_back(seedpoint);
		m_FirstSeedPointIndexList.push_back(m_nSeedPointNum - 1);
		m_isDrawing = true;
	}
	else
	{
		// clone the image which has been drew line
		m_SeedPointImgList.push_back(Utility::CloneImage(m_pImgTemp));
	}

	// Draw a point on image
	DrawPoint(m_SeedPointImgList.back(), seedpoint, point_r);
	PrintPicture(m_SeedPointImgList.back(), m_ImgPanel);

	CDialogEx::OnLButtonDown(nFlags, point);
}


void ProcessingDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default
	if(point.x < m_ImgRect.left || point.x > m_ImgRect.right || point.y < m_ImgRect.top || point.y > m_ImgRect.bottom)
		return;

	CPoint currentPoint = { point.x - m_ImgRect.left, point.y - m_ImgRect.top };
	char title[50];
	sprintf_s(title, "Processing  Point: (%d, %d)", currentPoint.x, currentPoint.y);
	SetWindowTextA(this->m_hWnd, title);
	
	if(m_isDrawing)
	{
		// clone the image which contains last seed point
		if(m_pImgTemp != nullptr)
			m_pImgTemp->Destroy();
		m_pImgTemp = Utility::CloneImage(m_SeedPointImgList.back());

		// get the shortest path from the seed point to current point
		vector<CPoint> tempPoints = m_Scissors->getShortestPathTo(currentPoint);

		DrawPointList(m_pImgTemp, tempPoints, 1);
		PrintPicture(m_pImgTemp, m_ImgPanel);
	}

	CDialogEx::OnMouseMove(nFlags, point);
}


BOOL ProcessingDlg::OnSetCursor(CWnd* pWnd, UINT nHitTest, UINT message)
{
	// TODO: Add your message handler code here and/or call default
	::SetCursor(LoadCursor(NULL, IDC_CROSS));;
	return TRUE;
}


void ProcessingDlg::OnRButtonDown(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default
	if(point.x < m_ImgRect.left || point.x > m_ImgRect.right || point.y < m_ImgRect.top || point.y > m_ImgRect.bottom)
		return;

	// destroy recent seed point and use last seed point
	if(m_nSeedPointNum > 0)
	{
		_cprintf("Delete last seed point.\n");
		if(m_isDrawing)
		{
			m_nSeedPointNum--;
			m_Scissors->removeSeedPoint();
		}

		m_isDrawing = m_nSeedPointNum != 0;

		// if last seed point is the first seed point of current curve, pop it from firstSeedPointList and stop drawing
		if(m_FirstSeedPointIndexList.back() == m_nSeedPointNum)
		{
			m_FirstSeedPointIndexList.pop_back();
			m_FirstSeedPointList.pop_back();
			m_isDrawing = false;
		}

		m_SeedPointImgList.back()->Destroy();
		m_SeedPointImgList.pop_back();

		ClearPicture();
		PrintPicture(m_SeedPointImgList.back(), m_ImgPanel);
	}
	
	CDialogEx::OnRButtonDown(nFlags, point);
}
