import os
import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

app = Flask(__name__, static_folder='static')

# ---------- 演示模式开关 ----------
DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'

# ---------- 全局变量 ----------
df = None
agent = None
llm = None

# ---------- 生成模拟数据 ----------
def generate_mock_data():
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    products = ['智能手表', '无线耳机', '平板电脑']
    data = []
    for date in dates:
        for product in products:
            base_sales = 5000 if product == '平板电脑' else 3000
            weekday_factor = 1.5 if date.weekday() >= 5 else 1.0
            sales = int(base_sales * weekday_factor * (0.8 + 0.4 * np.random.random()))
            cost_ratio = 0.6 if product == '平板电脑' else 0.5
            cost = sales * cost_ratio
            profit = sales - cost
            profit_margin = profit / sales if sales > 0 else 0
            data.append([date.strftime('%Y-%m-%d'), product, sales, cost, profit, profit_margin])
    df = pd.DataFrame(data, columns=['日期', '产品线', '销售额', '成本', '利润', '利润率'])
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/ops_data.csv', index=False)
    return df

# ---------- 加载初始数据 ----------
def load_initial_data():
    global df
    if not os.path.exists('data/ops_data.csv'):
        df = generate_mock_data()
    else:
        df = pd.read_csv('data/ops_data.csv')
        df['日期'] = pd.to_datetime(df['日期'])

# ---------- 创建 Agent ----------
def create_agent(dataframe):
    global llm
    llm = ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0
    )
    return create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        max_iterations=3,
    )

load_initial_data()
agent = create_agent(df)

def ask_llm(prompt: str) -> str:
    """真实调用大模型"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)

# ==================== 页面路由 ====================
@app.route('/')
def landing():
    """苹果风格首页"""
    return send_from_directory('static', 'index.html')

@app.route('/dashboard.html')
def dashboard_page():
    """仪表盘页面"""
    return send_from_directory('static', 'dashboard.html')

@app.route('/reports.html')
def reports_page():
    return send_from_directory('static', 'reports.html')

@app.route('/datasource.html')
def datasource_page():
    return send_from_directory('static', 'datasource.html')

@app.route('/settings.html')
def settings_page():
    return send_from_directory('static', 'settings.html')

@app.route('/health.html')
def health_page():
    return send_from_directory('static', 'health.html')

@app.route('/risks.html')
def risks_page():
    return send_from_directory('static', 'risks.html')

@app.route('/generate_report.html')
def generate_report_page():
    return send_from_directory('static', 'generate_report.html')

# ==================== 原有仪表盘 API ====================
@app.route('/api/kpi_overview')
def kpi_overview():
    global df
    try:
        total_sales = df['销售额'].sum()
        total_profit = df['利润'].sum()
        avg_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

        product_stats = df.groupby('产品线').agg({
            '销售额': 'sum',
            '利润': 'sum',
            '利润率': 'mean'
        }).reset_index()
        product_stats['销售额'] = product_stats['销售额'].astype(float)
        product_stats['利润'] = product_stats['利润'].astype(float)
        product_stats['利润率'] = product_stats['利润率'].astype(float)

        recent = df[df['日期'] >= df['日期'].max() - pd.Timedelta(days=30)]
        trend = recent.groupby('日期')[['销售额', '利润']].sum().reset_index()
        trend['日期'] = trend['日期'].dt.strftime('%Y-%m-%d')
        trend['销售额'] = trend['销售额'].astype(float)
        trend['利润'] = trend['利润'].astype(float)

        return jsonify({
            'kpi': {
                'total_sales': float(round(total_sales, 2)),
                'total_profit': float(round(total_profit, 2)),
                'avg_margin': float(round(avg_margin, 2))
            },
            'product_stats': product_stats.to_dict(orient='records'),
            'trend': {
                '日期': trend['日期'].tolist(),
                '销售额': trend['销售额'].tolist(),
                '利润': trend['利润'].tolist()
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_data():
    global df, agent
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': '未收到文件'}), 400

        csv_text = file.read().decode('utf-8')
        new_df = pd.read_csv(io.StringIO(csv_text))

        required_cols = ['日期', '产品线', '销售额', '成本', '利润', '利润率']
        if not all(col in new_df.columns for col in required_cols):
            return jsonify({'error': f'CSV 必须包含以下列：{required_cols}'}), 400

        new_df['日期'] = pd.to_datetime(new_df['日期'])
        df = new_df
        agent = create_agent(df)
        df.to_csv('data/ops_data.csv', index=False)

        return jsonify({'message': '数据上传成功，Agent 已更新'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def handle_query():
    global agent
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    try:
        prompt = f"""
        你是一个企业运营数据分析助手。请根据用户问题，使用提供的 DataFrame `df` 进行分析。
        要求：
        1. 如果需要计算，请写出 Pandas 代码并执行。
        2. 将最终结果整理成一个清晰的表格或数值。
        3. 在结果后面附上一段不超过100字的业务解读。
        用户问题：{question}
        """
        result = agent.invoke(prompt)
        output = result['output']
        chart_data = get_chart_data_for_question(question)
        return jsonify({'answer': output, 'chart_data': chart_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_chart_data_for_question(question):
    global df
    try:
        if '产品' in question or '利润' in question:
            product_profit = df.groupby('产品线')['利润'].sum().reset_index()
            return {
                'type': 'bar',
                'x': product_profit['产品线'].astype(str).tolist(),
                'y': product_profit['利润'].astype(float).tolist(),
                'title': '各产品线利润对比'
            }
        else:
            recent = df[df['日期'] >= df['日期'].max() - pd.Timedelta(days=30)]
            trend = recent.groupby('日期')[['销售额', '利润']].sum().reset_index()
            trend['日期'] = trend['日期'].dt.strftime('%Y-%m-%d')
            return {
                'type': 'line',
                'x': trend['日期'].tolist(),
                'series': [
                    {'name': '销售额', 'data': trend['销售额'].astype(float).tolist()},
                    {'name': '利润', 'data': trend['利润'].astype(float).tolist()}
                ],
                'title': '近30天销售与利润趋势'
            }
    except Exception as e:
        return {'error': str(e)}

# ==================== 模拟数据接口 ====================
@app.route('/api/reports')
def api_reports():
    reports = [
        {"id": 1, "date": "2026-04-16", "title": "4月16日运营日报", "summary": "销售额环比增长5.2%，平板电脑利润率提升明显。"},
        {"id": 2, "date": "2026-04-15", "title": "4月15日运营分析", "summary": "无线耳机销量下降，建议检查库存及促销策略。"},
        {"id": 3, "date": "2026-04-14", "title": "周度利润归因报告", "summary": "华东区退货率上升导致整体利润下滑2.1%。"},
    ]
    return jsonify(reports)

@app.route('/api/datasource/info')
def api_datasource_info():
    global df
    info = {
        "filename": "ops_data.csv",
        "rows": len(df),
        "columns": df.columns.tolist(),
        "last_modified": "2026-04-17 10:30:00"
    }
    return jsonify(info)

@app.route('/api/system/status')
def api_system_status():
    return jsonify({
        "model": "qwen-plus",
        "api_status": "connected",
        "version": "1.0.0",
        "last_update": "2026-04-18",
        "demo_mode": DEMO_MODE          # 新增：前端需要显示演示模式状态
    })

# ==================== 新功能 API（含演示模式） ====================
@app.route('/api/health_assessment')
def health_assessment():
    global df
    try:
        total_sales = df['销售额'].sum()
        total_profit = df['利润'].sum()
        avg_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

        max_date = df['日期'].max()
        last_month_start = max_date - pd.DateOffset(months=1)
        last_month_data = df[(df['日期'] >= last_month_start) & (df['日期'] < max_date)]
        current_month_data = df[df['日期'] >= max_date.replace(day=1)]

        last_sales = last_month_data['销售额'].sum()
        last_profit = last_month_data['利润'].sum()
        curr_sales = current_month_data['销售额'].sum()
        curr_profit = current_month_data['利润'].sum()

        sales_growth = ((curr_sales - last_sales) / last_sales * 100) if last_sales > 0 else 0
        profit_growth = ((curr_profit - last_profit) / last_profit * 100) if last_profit > 0 else 0

        def calc_score(value, thresholds):
            if value >= thresholds[1]: return 85 + min(15, (value - thresholds[1]) / (thresholds[1] * 0.5) * 15)
            if value >= thresholds[0]: return 60 + (value - thresholds[0]) / (thresholds[1] - thresholds[0]) * 25
            return max(0, value / thresholds[0] * 60)

        profit_score = calc_score(avg_margin, [20, 40])
        sales_score = calc_score(sales_growth, [5, 15])
        profit_growth_score = calc_score(profit_growth, [5, 15])
        total_score = int(profit_score * 0.4 + sales_score * 0.3 + profit_growth_score * 0.3)

        if DEMO_MODE:
            ai_comment = """【演示模式】整体运营状况良好，综合健康度得分85分。盈利能力表现突出，平均利润率稳定在45%以上；增长潜力方面，销售额环比增长8.5%，保持稳健上升趋势。建议继续优化高利润产品线（如平板电脑）的营销策略，同时关注无线耳机的库存周转情况。"""
        else:
            prompt = f"""
            你是一位资深企业战略顾问。请基于以下运营数据，提供一份简练的企业健康度评估（不超过200字）。
            - 综合健康度得分：{total_score}/100
            - 盈利能力得分：{profit_score:.1f}/100（当前平均利润率 {avg_margin:.2f}%）
            - 增长潜力得分：{sales_score:.1f}/100（销售额环比 {sales_growth:+.2f}%）
            - 利润增长得分：{profit_growth_score:.1f}/100（利润环比 {profit_growth:+.2f}%）
            请完成：
            1. 一句话总结整体健康状况。
            2. 指出最突出的优点和风险点。
            3. 提供一条具体改进建议。
            """
            ai_comment = ask_llm(prompt).strip()

        return jsonify({
            'total_score': total_score,
            'dimensions': {
                'profitability': round(profit_score, 1),
                'growth_potential': round(sales_score, 1),
                'profit_growth': round(profit_growth_score, 1)
            },
            'metrics': {
                'avg_margin': round(avg_margin, 2),
                'sales_growth': round(sales_growth, 2),
                'profit_growth': round(profit_growth, 2)
            },
            'ai_comment': ai_comment
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk_opportunity')
def risk_opportunity():
    global df
    try:
        risks = []
        opportunities = []

        max_date = df['日期'].max()
        last_week_start = max_date - timedelta(days=7)
        two_weeks_ago_start = max_date - timedelta(days=14)

        current_week = df[df['日期'] >= last_week_start]
        previous_week = df[(df['日期'] >= two_weeks_ago_start) & (df['日期'] < last_week_start)]

        curr_by_product = current_week.groupby('产品线').agg({'销售额':'sum', '利润':'sum', '利润率':'mean'}).reset_index()
        prev_by_product = previous_week.groupby('产品线').agg({'销售额':'sum', '利润':'sum', '利润率':'mean'}).reset_index()

        for _, row in curr_by_product.iterrows():
            product = row['产品线']
            prev_row = prev_by_product[prev_by_product['产品线'] == product]
            if prev_row.empty:
                continue
            sales_change = (row['销售额'] - prev_row.iloc[0]['销售额']) / prev_row.iloc[0]['销售额'] * 100
            margin_change = row['利润率'] - prev_row.iloc[0]['利润率']
            if sales_change < -10:
                risks.append(f"⚠️ {product} 周销售额下降 {abs(sales_change):.1f}%")
            if margin_change < -0.05:
                risks.append(f"⚠️ {product} 利润率下降 {margin_change*100:.1f} 个百分点")

        for _, row in curr_by_product.iterrows():
            product = row['产品线']
            prev_row = prev_by_product[prev_by_product['产品线'] == product]
            if prev_row.empty:
                continue
            sales_change = (row['销售额'] - prev_row.iloc[0]['销售额']) / prev_row.iloc[0]['销售额'] * 100
            margin_change = row['利润率'] - prev_row.iloc[0]['利润率']
            if sales_change > 15:
                opportunities.append(f"🚀 {product} 周销售额增长 {sales_change:.1f}%")
            if margin_change > 0.05:
                opportunities.append(f"🚀 {product} 利润率提升 {margin_change*100:.1f} 个百分点")

        if not risks:
            risks.append("✅ 未检测到明显风险信号，各项指标平稳")
        if not opportunities:
            opportunities.append("📊 未检测到突出机会，建议持续关注高利润产品")

        if DEMO_MODE:
            ai_suggestion = """【演示模式】当前最值得关注的信号是平板电脑利润率提升5.2个百分点，显示高端产品线盈利能力增强。建议加大该产品线的营销投入，同时警惕无线耳机销售额增速放缓的风险，可考虑推出促销活动以维持市场份额。"""
        else:
            prompt = f"""
            你是一个敏锐的商业分析师。请根据以下自动识别的信号，提供不超过150字的优先级建议。
            风险信号：{'; '.join(risks)}
            机会信号：{'; '.join(opportunities)}
            请说明哪个信号最值得关注，并给出一个简单的行动方向。
            """
            ai_suggestion = ask_llm(prompt).strip()

        return jsonify({
            'risks': risks,
            'opportunities': opportunities,
            'ai_suggestion': ai_suggestion
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_report')
def generate_report():
    global df
    try:
        total_sales = df['销售额'].sum()
        total_profit = df['利润'].sum()
        avg_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

        max_date = df['日期'].max()
        last_month_data = df[(df['日期'] >= max_date - pd.DateOffset(months=1)) & (df['日期'] < max_date)]
        curr_month_data = df[df['日期'] >= max_date.replace(day=1)]
        sales_growth = ((curr_month_data['销售额'].sum() - last_month_data['销售额'].sum()) / last_month_data['销售额'].sum() * 100) if last_month_data['销售额'].sum() > 0 else 0

        product_profit = df.groupby('产品线')['利润'].sum().sort_values(ascending=False)
        top_product = product_profit.index[0] if not product_profit.empty else '无'

        if DEMO_MODE:
            report_content = f"""# 企业运营分析报告（演示模式）

**生成时间**：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、核心摘要
本月整体运营表现稳健，总销售额达到 ¥{total_sales:,.0f}，环比增长 {sales_growth:+.1f}%。利润最高的产品线为 {top_product}，贡献了主要利润增长点。系统健康度评分为 85 分，处于良好水平。

## 二、KPI 概览
| 指标 | 数值 |
|------|------|
| 总销售额 | ¥{total_sales:,.2f} |
| 总利润 | ¥{total_profit:,.2f} |
| 平均利润率 | {avg_margin:.2f}% |
| 月度销售额环比 | {sales_growth:+.2f}% |

## 三、产品线表现
- **{top_product}**：利润贡献最高，利润率稳定，建议持续投入。
- **智能手表**：销售额平稳，可作为稳定现金流来源。
- **无线耳机**：销量略有下滑，需关注市场竞争动态。

## 四、结论与建议
1. 加大对 {top_product} 的营销资源倾斜，巩固优势地位。
2. 针对无线耳机开展限时促销，防止市场份额进一步流失。
3. 持续监控整体利润率变化，控制成本支出。

*（注：本报告为演示模式下的模拟内容，实际调用 AI 将生成更详细的分析。）*
"""
        else:
            prompt = f"""
            你是一位专业的企业运营分析师。请根据以下数据，生成一份《企业运营分析报告》（Markdown格式），要求结构清晰、语言专业，字数500字以内。
            核心数据：
            - 总销售额：¥{total_sales:,.2f}
            - 总利润：¥{total_profit:,.2f}
            - 平均利润率：{avg_margin:.2f}%
            - 月度销售额环比：{sales_growth:+.2f}%
            - 利润最高产品线：{top_product}

            报告结构要求：
            1. 核心摘要（一段话概括本月表现）
            2. KPI概览（关键指标表格）
            3. 产品线表现（简要分析）
            4. 结论与建议（1-2条可执行建议）
            """
            report_content = ask_llm(prompt).strip()

        return jsonify({
            'report': report_content,
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    @app.route('/static/<path:filename>')
    def static_files(filename):
        response = send_from_directory('static', filename)
        response.headers['Cache-Control'] = 'public, max-age=86400'  
        return response

if __name__ == '__main__':
    import webbrowser
    import threading
    
    # 启动 Flask 后自动打开浏览器
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:5000')
    
    # 延迟 1.5 秒确保服务器完全启动
    threading.Timer(1.5, open_browser).start()
    
    # 启动 Flask 应用（关闭 debug 模式避免重复打开）
    app.run(debug=False, port=5000)