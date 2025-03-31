# 🛡️ Security Incidents: A Data Story  
## 安全事件的数据叙事分析

This project presents an interactive narrative analysis of global security incidents from 2000 to 2023. Using Python, we perform data cleaning, visualization, regression and classification to uncover hidden patterns in incident data. The analysis is published as a scrollable story webpage via GitHub Pages.

本项目通过 Python 对 2000–2023 年的全球安全事件数据进行探索性分析、回归建模和分类建模，最终以可视化图 + 分析解说的方式发布在 GitHub Pages 上，形成一个可滚动的叙事型网页。

---

## 🔍 Project Highlights | 项目亮点

- Analyze temporal trends in incident frequency  
  分析事件发生的时间趋势
- Understand casualty metrics and their relationships  
  研究伤亡相关指标之间的关系
- Build linear regression and logistic regression models  
  构建线性回归与逻辑回归模型
- Apply Random Forest for classification of verified cases  
  使用随机森林分类“是否被验证”的事件

---

## 🌐 Live Website | 在线演示网页

👉 [**Click to View the Web Report**](https://yourusername.github.io/security-analysis/)  
（请替换成你自己的 GitHub Pages 链接）

---

## 📊 Structure of the Visual Story | 可视化内容结构

1. 📈 Yearly & Monthly Trends
2. ⚠️ Casualty Metric Distributions
3. 🔍 Scatter Matrix & Feature Correlation
4. 🧮 Linear Regression: Predicting Total Affected
5. ✅ Classification (Logistic + Random Forest)
6. 📌 Key Findings & Limitations

Each section contains one figure and one explanation block to form a clean story.

每一部分都包含“一张图 + 一段解释”，构成完整的数据故事。

---

## 🧠 Technologies Used | 使用技术

- Python (pandas, numpy, plotly, sklearn)
- HTML + CSS (for visual storytelling layout)
- GitHub Pages (for free web publishing)

---

## 📁 File Structure | 文件结构说明

```
security-analysis/
├── index.html              # Main storytelling webpage
├── assets/
│   ├── figures/            # All visual charts (e.g., fig1.png, fig2.png ...)
│   └── style.css           # Custom style sheet
├── scripts/
│   └── interaactive.py     # Main Python analysis script
└── README.md               # This file
```

---

## 🚀 Run the Python Analysis | 如何运行代码分析

To reproduce the analysis and charts locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/security-analysis.git
   cd security-analysis/scripts
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy plotly scikit-learn
   ```

3. Run the script:
   ```bash
   python interaactive.py
   ```

Interactive figures will open in your browser automatically.

---

## ✍️ Author | 作者

KeHan Wu  
March 2025  
Georgetown University - DSAN Program

欢迎交流或提出建议！
