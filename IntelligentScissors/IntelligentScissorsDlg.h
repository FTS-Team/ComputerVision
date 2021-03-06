
// IntelligentScissorsDlg.h : header file
//

#pragma once
#include "ProcessingDlg.h"

// CIntelligentScissorsDlg dialog
class CIntelligentScissorsDlg : public CDialogEx
{
// Construction
public:
	CIntelligentScissorsDlg(CWnd* pParent = nullptr);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_INTELLIGENTSCISSORS_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

protected:
	void PrintPicture(CImage *pImgSrc, CStatic &cPicPanel);
	void ClearPicture();

public:
	afx_msg void OnBnClickedButtonOpenimage();
	afx_msg void OnBnClickedButtonProcess();
	afx_msg void OnCbnSelchangeComboImageShowType();

private:
	CImage * m_pImgSrc;
	CString m_strImgPath;
	CStatic m_ImgPanel;
	CComboBox m_ComboImageShowType;
};
