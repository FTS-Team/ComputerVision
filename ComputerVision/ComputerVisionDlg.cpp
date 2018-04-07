
// ComputerVisionDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "ComputerVision.h"
#include "ComputerVisionDlg.h"
#include "afxdialogex.h"


#include "ImageMatch.h"


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
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

}

void CComputerVisionDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CComputerVisionDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_OPEN, &CComputerVisionDlg::OnBnClickedButtonOpen)
	ON_BN_CLICKED(IDC_BUTTON_PROCESS, &CComputerVisionDlg::OnBnClickedButtonProcess)
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

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标




	// TODO: 在此添加额外的初始化代码

	//初始化图片选择下拉框
	CComboBox * cmb_picture = ((CComboBox*)GetDlgItem(IDC_COMBO_PICTURE));
	cmb_picture->AddString(_T("图片1"));
	cmb_picture->AddString(_T("图片2"));
	cmb_picture->SetCurSel(0);//图片1

	//初始化算法选择下拉框
	CComboBox * cmb_match = ((CComboBox*)GetDlgItem(IDC_COMBO_MATCH));
	cmb_match->AddString(_T("SIFT"));
	cmb_match->AddString(_T("SURF"));
	cmb_match->AddString(_T("ORB"));
	cmb_match->AddString(_T("FERNS"));
	cmb_match->SetCurSel(0);


	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CComputerVisionDlg::OnSysCommand(UINT nID, LPARAM lParam)
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


//绘制图片到图片框
void CComputerVisionDlg::drawImage(CWnd *pWnd, CImage& imgSrc) {
	int height;
	int width;
	CRect rect;
	CRect rect1;
	height = imgSrc.GetHeight();
	width = imgSrc.GetWidth();

	pWnd->GetClientRect(&rect);//获取客户区域矩形


	CDC *pDC = pWnd->GetDC();//获取静态框句柄
	SetStretchBltMode(pDC->m_hDC, STRETCH_HALFTONE);



	if (width <= rect.Width() && height <= rect.Height())
	{
		rect1 = CRect(rect.TopLeft(), CSize(width, height));
		rect1.left += 1;
		rect1.bottom -= 1;
		rect1.right -= 1;
		rect1.top += 1;
		imgSrc.StretchBlt(pDC->m_hDC, rect1, SRCCOPY);//图片显示
	}
	else
	{

		float xScale = (float)rect.Width() / (float)width;
		float yScale = (float)rect.Height() / (float)height;
		float ScaleIndex = (xScale <= yScale ? xScale : yScale);
		rect1 = CRect(rect.TopLeft(), CSize((int)width*ScaleIndex, (int)height*ScaleIndex));//拉伸
		rect1.left += 1;
		rect1.bottom -= 1;
		rect1.right -= 1;
		rect1.top += 1;

		imgSrc.StretchBlt(pDC->m_hDC, rect1, SRCCOPY);

	}
	ReleaseDC(pDC);//释放句柄
}



// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CComputerVisionDlg::OnPaint()
{
	if (IsIconic())
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
	}
	else
	{
		CDialogEx::OnPaint();

		CWnd *pWnd;

		if (!imgSrc1.IsNull())
		{
			pWnd = GetDlgItem(IDC_PICTURE1);
			drawImage(pWnd, imgSrc1);
		}
		if (!imgSrc2.IsNull()) {
			pWnd = GetDlgItem(IDC_PICTURE2);
			drawImage(pWnd, imgSrc2);
		}

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
	TCHAR szFilter[] = _T("JPEG(*jpg)|*.jpg|*.bmp|*.png|TIFF(*.tif)|*.tif|All Files （*.*）|*.*||");
	CString filePath("");

	CFileDialog fileOpenDialog(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);//初始化打开文件对话框
	if (fileOpenDialog.DoModal() == IDOK)//确保正确
	{
		VERIFY(filePath = fileOpenDialog.GetPathName());//断言并获取非空路径名


		if (((CComboBox*)GetDlgItem(IDC_COMBO_PICTURE))->GetCurSel() == 0) {//图片1
			imgSrc1.Destroy();
			imgSrc1.Load(filePath);//加载图片1
			output(this, IDC_EDIT_OUT, _T("加载图片1 "));
			
		}
		else {
			imgSrc2.Destroy();
			imgSrc2.Load(filePath);//加载图片2
			output(this, IDC_EDIT_OUT, _T("加载图片2"));
		}
		this->Invalidate();//更新绘图Paint

	}
}



//处理图片事件
void CComputerVisionDlg::OnBnClickedButtonProcess()
{
	//图片为空
	if (imgSrc1.IsNull() || imgSrc2.IsNull()) {
		output(this, IDC_EDIT_OUT, _T("图片1或图片2为空"));
		return;
	}
	
	CComboBox * cmb_match = ((CComboBox*)GetDlgItem(IDC_COMBO_MATCH));
	int match = cmb_match->GetCurSel();

	switch (match)
	{
	case Match::SIFT:
	{
		output(this, IDC_EDIT_OUT, _T("开始进行基于SIFT算法的图像匹配处理"));
		ImageMatch::SIFT(imgSrc1, imgSrc2);

	}
	break;


	case Match::SURF:
	{
		output(this, IDC_EDIT_OUT, _T("开始进行基于SURF算法的图像匹配处理"));
		ImageMatch::SURF(imgSrc1, imgSrc2);

	}
	break;

	case Match::ORB:
	{
		output(this, IDC_EDIT_OUT, _T("开始进行基于ORB算法的图像匹配处理"));
		ImageMatch::ORBMatch(imgSrc1, imgSrc2);
	}
	break;


	case Match::FERNS:


	default:
		break;
	}

	output(this, IDC_EDIT_OUT, _T("处理完成"));
}


//添加输出框的值
bool CComputerVisionDlg::output(LPVOID p, UINT id, CString str) {

	CComputerVisionDlg * dlg = (CComputerVisionDlg *)p;
	CEdit * outEdit = (CEdit*)dlg->GetDlgItem(id);//输出框

	int nLength = outEdit->SendMessage(WM_GETTEXTLENGTH);
	outEdit->SetSel(nLength, nLength);
	outEdit->ReplaceSel(_T(">>>")+str+"\n");

	return true;

}