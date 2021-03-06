
// ComputerVisionDlg.h: 头文件
//

#pragma once


// CComputerVisionDlg 对话框
class CComputerVisionDlg : public CDialogEx
{
// 构造
public:
	CComputerVisionDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_COMPUTERVISION_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	//数据成员
	//CImage imgSrc1;//存储图片1
	//CImage imgSrc2;//存储图片2



	//成员函数
	bool CComputerVisionDlg::OutputLog(CString str);//添加输出框的值
	void PrintPicture(CImage *pImgSrc, CStatic &cPicPanel);
	void ClearPicture();

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:



	afx_msg void OnBnClickedButtonOpen();
	afx_msg void OnBnClickedButtonProcess();
	afx_msg void OnBnClickedRadioPitcure1();
	afx_msg void OnBnClickedRadioPitcure2();
	afx_msg void OnBnClickedRadioSift();
	afx_msg void OnBnClickedRadioSurf();
	afx_msg void OnBnClickedRadioOrb();
	afx_msg void OnBnClickedRadioFerns();

private:
	int m_nRadioPicture;
	int m_nRadioAlgorithm;
	CImage *m_pImgSrc[2];
	CString m_strImgPath[2];
	CStatic m_Picture[2];
};
