
// IntelligentScissorsDlg.cpp : implementation file
//

#include "stdafx.h"
#include "IntelligentScissors.h"
#include "IntelligentScissorsDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CIntelligentScissorsDlg dialog



CIntelligentScissorsDlg::CIntelligentScissorsDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_INTELLIGENTSCISSORS_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CIntelligentScissorsDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_STATIC_IMGPANEL, m_ImgPanel);
}

BEGIN_MESSAGE_MAP(CIntelligentScissorsDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_OPENIMAGE, &CIntelligentScissorsDlg::OnBnClickedButtonOpenimage)
	ON_BN_CLICKED(IDC_BUTTON_PROCESS, &CIntelligentScissorsDlg::OnBnClickedButtonProcess)
END_MESSAGE_MAP()


// CIntelligentScissorsDlg message handlers

BOOL CIntelligentScissorsDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CIntelligentScissorsDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CIntelligentScissorsDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CIntelligentScissorsDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CIntelligentScissorsDlg::PrintPicture(CImage * pImgSrc, CStatic & cPicPanel)
{
	if(pImgSrc != NULL)
	{
		int height;
		int width;
		CRect rect;
		CRect rect1;
		height = pImgSrc->GetHeight();
		width = pImgSrc->GetWidth();

		cPicPanel.GetClientRect(&rect);
		int rect_width = rect.Width() - 4;
		int rect_height = rect.Height() - 4;

		CDC *pDC = cPicPanel.GetDC();
		SetStretchBltMode(pDC->m_hDC, STRETCH_HALFTONE);

		if(width > rect_width || height > rect_height)
		{
			float xScale = (float) rect_width / (float) width;
			float yScale = (float) rect_height / (float) height;
			float ScaleIndex = (xScale <= yScale ? xScale : yScale);
			width *= ScaleIndex;
			height *= ScaleIndex;
		}

		rect1 = CRect(
			CPoint((int) ((rect.Width() - width) / 2), (int) ((rect.Height() - height) / 2)),
			CSize((int) width, (int) height)
		);
		pImgSrc->StretchBlt(pDC->m_hDC, rect1, SRCCOPY);
		ReleaseDC(pDC);
	}
}

void CIntelligentScissorsDlg::ClearPicture()
{
	CRect rect;
	m_ImgPanel.GetClientRect(&rect);
	m_ImgPanel.GetDC()->FillSolidRect(rect.left + 1, rect.top + 1, rect.Width() - 2, rect.Height() - 2, RGB(240, 240, 240));
}



void CIntelligentScissorsDlg::OnBnClickedButtonOpenimage()
{
	// TODO: Add your control notification handler code here
	TCHAR szFilter[] = _T("JPEG (*.jpg)|*.jpg|BMP (*.bmp)|*.bmp|PNG (*.png)|*.png|TIFF (*.tif)|*.tif|All Files（*.*）|*.*||");
	CString filePath("");

	CFileDialog fileOpenDialog(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);
	if(fileOpenDialog.DoModal() == IDOK)
	{
		VERIFY(filePath = fileOpenDialog.GetPathName());

		m_strImgPath = filePath; 
		((CEdit*) GetDlgItem(IDC_EDIT_IMGNAME))->SetWindowTextW(m_strImgPath.Right(m_strImgPath.GetLength() - m_strImgPath.ReverseFind('\\') - 1));

		if(m_pImgSrc != NULL)
		{
			m_pImgSrc->Destroy();
			delete m_pImgSrc;
		}
		m_pImgSrc = new CImage();
		m_pImgSrc->Load(filePath);

		// 先清空原来的图片，然后显示新的图片
		ClearPicture();
		PrintPicture(m_pImgSrc, m_ImgPanel);
	}
}


void CIntelligentScissorsDlg::OnBnClickedButtonProcess()
{
	// TODO: Add your control notification handler code here
}
