
// ComputerVisionDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "ComputerVision.h"
#include "ComputerVisionDlg.h"
#include "afxdialogex.h"


#include "ImageMatch.h"
#include <io.h>    
#include <fcntl.h> 
#include <conio.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
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


// CComputerVisionDlg 对话框



CComputerVisionDlg::CComputerVisionDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_COMPUTERVISION_DIALOG, pParent)
	, m_nRadioPicture(0)
	, m_nRadioAlgorithm(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	if(AllocConsole())
		_cprintf("Use console to display infomation in runtime.\n");
	putchar('.');
}

void CComputerVisionDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PICTURE1, m_Picture[0]);
	DDX_Control(pDX, IDC_PICTURE2, m_Picture[1]);
}

BEGIN_MESSAGE_MAP(CComputerVisionDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_OPEN, &CComputerVisionDlg::OnBnClickedButtonOpen)
	ON_BN_CLICKED(IDC_BUTTON_PROCESS, &CComputerVisionDlg::OnBnClickedButtonProcess)
	ON_BN_CLICKED(IDC_RADIO_PITCURE1, &CComputerVisionDlg::OnBnClickedRadioPitcure1)
	ON_BN_CLICKED(IDC_RADIO_PITCURE2, &CComputerVisionDlg::OnBnClickedRadioPitcure2)
	ON_BN_CLICKED(IDC_RADIO_SIFT, &CComputerVisionDlg::OnBnClickedRadioSift)
	ON_BN_CLICKED(IDC_RADIO_SURF, &CComputerVisionDlg::OnBnClickedRadioSurf)
	ON_BN_CLICKED(IDC_RADIO_ORB, &CComputerVisionDlg::OnBnClickedRadioOrb)
	ON_BN_CLICKED(IDC_RADIO_FERNS, &CComputerVisionDlg::OnBnClickedRadioFerns)
END_MESSAGE_MAP()


// CComputerVisionDlg 消息处理程序
BOOL CComputerVisionDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if(pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if(!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	CButton* radio_picture = (CButton*) GetDlgItem(IDC_RADIO_PITCURE1);
	radio_picture->SetCheck(1);
	CButton* radio_algorithm = (CButton*) GetDlgItem(IDC_RADIO_SIFT);
	radio_algorithm->SetCheck(1);

	////初始化图片选择下拉框
	//CComboBox * cmb_picture = ((CComboBox*)GetDlgItem(IDC_COMBO_PICTURE));
	//cmb_picture->AddString(_T("图片1"));
	//cmb_picture->AddString(_T("图片2"));
	//cmb_picture->SetCurSel(0);//图片1

	////初始化算法选择下拉框
	//CComboBox * cmb_match = ((CComboBox*)GetDlgItem(IDC_COMBO_MATCH));
	//cmb_match->AddString(_T("SIFT"));
	//cmb_match->AddString(_T("SURF"));
	//cmb_match->AddString(_T("ORB"));
	//cmb_match->AddString(_T("FERNS"));
	//cmb_match->SetCurSel(0);


	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CComputerVisionDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	} else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}



// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CComputerVisionDlg::OnPaint()
{
	if(IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	} else
	{
		CDialogEx::OnPaint();

		/*CWnd *pWnd;

		if (!imgSrc1.IsNull())
		{
			pWnd = GetDlgItem(IDC_PICTURE1);
			drawImage(pWnd, imgSrc1);
		}
		if (!imgSrc2.IsNull()) {
			pWnd = GetDlgItem(IDC_PICTURE2);
			drawImage(pWnd, imgSrc2);
		}*/

	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CComputerVisionDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


//打开图片事件
void CComputerVisionDlg::OnBnClickedButtonOpen()
{
	// TODO: 在此添加控件通知处理程序代码
	TCHAR szFilter[] = _T("JPEG (*.jpg)|*.jpg|BMP (*.bmp)|*.bmp|PNG (*.png)|*.png|TIFF (*.tif)|*.tif|All Files（*.*）|*.*||");
	CString filePath("");

	CFileDialog fileOpenDialog(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);//初始化打开文件对话框
	if(fileOpenDialog.DoModal() == IDOK)//确保正确
	{
		VERIFY(filePath = fileOpenDialog.GetPathName());//断言并获取非空路径名

		m_strImgPath[m_nRadioPicture] = filePath; //记录图片路径

		// 打开图片
		if(m_pImgSrc[m_nRadioPicture] != NULL)
		{
			m_pImgSrc[m_nRadioPicture]->Destroy();
			delete m_pImgSrc[m_nRadioPicture];
		}
		m_pImgSrc[m_nRadioPicture] = new CImage();
		m_pImgSrc[m_nRadioPicture]->Load(filePath);

		// 先清空原来的图片，然后显示新的图片
		ClearPicture();
		PrintPicture(m_pImgSrc[m_nRadioPicture], m_Picture[m_nRadioPicture]);
	}
}



//处理图片事件
void CComputerVisionDlg::OnBnClickedButtonProcess()
{
	//图片为空
	if(m_pImgSrc[0] == nullptr)
	{
		OutputLog(_T("图片1为空"));
		return;
	}
	if(m_pImgSrc[1] == nullptr)
	{
		OutputLog(_T("图片2为空"));
		return;
	}


	switch(m_nRadioAlgorithm)
	{
		case Match::SIFT:
			OutputLog(_T("开始进行基于SIFT算法的图像匹配处理"));
			ImageMatch::SIFT(*m_pImgSrc[0], *m_pImgSrc[1]);
			break;
		case Match::SURF:
			OutputLog(_T("开始进行基于SURF算法的图像匹配处理"));
			ImageMatch::SURF(*m_pImgSrc[0], *m_pImgSrc[1]);
			break;
		case Match::ORB:
			OutputLog(_T("开始进行基于ORB算法的图像匹配处理"));
			ImageMatch::ORBMatch(*m_pImgSrc[0], *m_pImgSrc[1]);
			break;
		case Match::FERNS:
			OutputLog(_T("开始进行基于FERNS算法的图像匹配处理"));
			ImageMatch::FERNS(m_strImgPath[0], m_strImgPath[1]);
			break;
		default:
			break;
	}


}

// 添加输出框的值
bool CComputerVisionDlg::OutputLog(CString str)
{
	// 输出框
	CEdit * outEdit = (CEdit*) ((CComputerVisionDlg *) this)->GetDlgItem(IDC_EDIT_OUT);

	int nLength = outEdit->SendMessage(WM_GETTEXTLENGTH);
	outEdit->SetSel(nLength, nLength);
	outEdit->ReplaceSel(_T(">>> ") + str + "\n");

	return true;
}

void CComputerVisionDlg::PrintPicture(CImage * pImgSrc, CStatic & cPicPanel)
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

void CComputerVisionDlg::ClearPicture()
{
	CRect rect;
	m_Picture[m_nRadioPicture].GetClientRect(&rect);
	m_Picture[m_nRadioPicture].GetDC()->FillSolidRect(rect.left + 1, rect.top + 1, rect.Width() - 2, rect.Height() - 2, RGB(240, 240, 240));
}


void CComputerVisionDlg::OnBnClickedRadioPitcure1()
{
	// TODO: Add your control notification handler code here
	m_nRadioPicture = 0;
}


void CComputerVisionDlg::OnBnClickedRadioPitcure2()
{
	// TODO: Add your control notification handler code here
	m_nRadioPicture = 1;
}


void CComputerVisionDlg::OnBnClickedRadioSift()
{
	// TODO: Add your control notification handler code here
	m_nRadioAlgorithm = 0;
}


void CComputerVisionDlg::OnBnClickedRadioSurf()
{
	// TODO: Add your control notification handler code here
	m_nRadioAlgorithm = 1;
}


void CComputerVisionDlg::OnBnClickedRadioOrb()
{
	// TODO: Add your control notification handler code here
	m_nRadioAlgorithm = 2;
}


void CComputerVisionDlg::OnBnClickedRadioFerns()
{
	// TODO: Add your control notification handler code here
	m_nRadioAlgorithm = 3;
}
