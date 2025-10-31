import streamlit as st
import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import wave
import pyaudio
import time
from datetime import datetime
import matplotlib.pyplot as plt
import io

# 设置页面
st.set_page_config(
    page_title="支气管炎风险检测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 检查pyaudio是否可用
try:
    import pyaudio
    pyaudio_available = True
except ImportError:
    pyaudio_available = False
    st.warning("⚠️ 录音功能不可用，请安装pyaudio包")

# 标题和介绍
st.title("🏥 云端肺健康——基于语音的支气管炎远程筛查系统")
st.markdown("""
该系统通过分析您的呼吸声或语音，评估您患支气管炎的风险概率。
请按照以下步骤进行操作：
""")
st.info("""
**温馨提示：**
-请确保在安静的环境下录音，避免背景噪音影响结果。
-录音时请保持正常呼吸，距离麦克风10-15厘米。
-建议录制3-10秒的清晰音频。
-本系统仅为辅助评估工具，不能替代专业医疗诊断，如有不适请及时就医。
""")

# 检查模型是否存在
@st.cache_resource
def load_prediction_model():
    try:
        model = load_model("models/bronchitis_model.h5")
        label_encoder = np.load("models/label_encoder.npy", allow_pickle=True)
        return model, label_encoder
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None, None

# 直接使用原有的预测代码
class BronchitisPredictor:
    def __init__(self, model_path="models/bronchitis_model.h5"):
        try:
            self.model = load_model(model_path)
            self.label_encoder = np.load("models/label_encoder.npy", allow_pickle=True)
            self.max_pad_len = 174
        except Exception as e:
            st.error(f"初始化预测器失败: {str(e)}")
            raise e

    def extract_features(self, audio_path):
        try:
            audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            pad_width = self.max_pad_len - mfccs.shape[1]
            if pad_width < 0:
                mfccs = mfccs[:, :self.max_pad_len]
            else:
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            return mfccs
        except Exception as e:
            st.error(f"处理音频时出错: {str(e)}")
            return None

    def predict(self, audio_path):
        features = self.extract_features(audio_path)
        if features is None:
            return "Error: 无法处理音频文件", 0.0

        features = features[np.newaxis, ..., np.newaxis]
        prediction = self.model.predict(features, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = self.label_encoder[predicted_index]
        confidence = np.max(prediction)

        # 计算支气管炎风险概率
        bronchitis_prob = 0.0
        if predicted_label == "bronchitis":
            bronchitis_prob = confidence
        elif predicted_label == "healthy_breath":
            bronchitis_prob = 1 - confidence
        elif predicted_label == "healthy_voice":
            # 语音样本的支气管炎风险较低
            bronchitis_idx = np.where(self.label_encoder == "bronchitis")[0]
            if len(bronchitis_idx) > 0:
                bronchitis_prob = prediction[0][bronchitis_idx[0]]

        return predicted_label, bronchitis_prob

# 录音功能 - 基于原有代码
def record_audio(filename, duration=5, sample_rate=44100):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    frames = []
    
    for _ in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # 保存录音
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return True

# 分析音频并显示结果
def analyze_audio(audio_path):
    try:
        predictor = BronchitisPredictor()
        label, risk = predictor.predict(audio_path)
        
        # 显示结果
        st.subheader("分析结果")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("音频类型", label.replace('_', ' ').title())
        
        with col2:
            st.metric("支气管炎风险概率", f"{risk:.2%}")
        
        # 风险评估
        if risk > 0.7:
            st.error("评估: 支气管炎高风险")
            st.info("建议: 请立即咨询医疗专业人士")
        elif risk > 0.4:
            st.warning("评估: 支气管炎中度风险")
            st.info("建议: 监测症状并考虑医疗咨询")
        else:
            st.success("评估: 支气管炎低风险")
            st.info("建议: 未检测到立即关注的问题")
        
        # 风险可视化
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['风险等级'], [risk], color='skyblue', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('风险概率')
        ax.axvline(x=0.4, color='orange', linestyle='--', label='中度风险阈值')
        ax.axvline(x=0.7, color='red', linestyle='--', label='高风险阈值')
        ax.legend()
        
        st.pyplot(fig)
        
        st.markdown("---")
        
    except Exception as e:
        st.error(f"分析过程中出错: {str(e)}")

# 主应用逻辑
def main():
    # 侧边栏
    st.sidebar.title("导航")
    
    # 根据pyaudio可用性显示选项
    if pyaudio_available:
        app_mode = st.sidebar.radio("选择功能", 
                                   ["录音分析", "上传音频分析", "系统信息"])
    else:
        app_mode = st.sidebar.radio("选择功能", 
                                   ["上传音频分析", "系统信息"])
        st.sidebar.warning("录音功能当前不可用")
    
    # 创建临时目录
    if not os.path.exists("temp_audio"):
        os.makedirs("temp_audio")
    
    if app_mode == "录音分析" and pyaudio_available:
        st.header("录音分析")
        st.markdown("请录制您的呼吸声或语音进行支气管炎风险分析")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            duration = st.slider("录音时长(秒)", min_value=3, max_value=10, value=5)
            record_button = st.button("开始录音")
        
        with col2:
            if record_button:
                with st.spinner(f"录音中... 请保持安静并准备好录音 {duration} 秒"):
                    filename = f"temp_audio/recording_{int(time.time())}.wav"
                    try:
                        record_audio(filename, duration=duration)
                        st.success("录音完成!")
                        
                        # 播放录音
                        st.audio(filename, format="audio/wav")
                        
                        # 分析录音
                        analyze_audio(filename)
                        
                        # 可选：删除录音文件
                        if st.checkbox("分析完成后删除录音文件"):
                            os.remove(filename)
                            st.info("录音文件已删除")
                    except Exception as e:
                        st.error(f"录音失败: {str(e)}")
    
    elif app_mode == "上传音频分析":
        st.header("上传音频分析")
        st.markdown("请上传您的音频文件进行支气管炎风险分析")
        
        uploaded_file = st.file_uploader("选择音频文件", type=['wav', 'mp3', 'm4a', 'flac'])
        
        if uploaded_file is not None:
            # 保存上传的文件
            filename = f"temp_audio/uploaded_{int(time.time())}_{uploaded_file.name}"
            with open(filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("文件上传成功!")
            st.audio(filename, format="audio/wav")
            
            # 分析音频
            analyze_audio(filename)
            
            # 可选：删除上传的文件
            if st.checkbox("分析完成后删除上传的文件"):
                os.remove(filename)
                st.info("上传的文件已删除")
    
    elif app_mode == "系统信息":
        st.header("系统信息")
        st.markdown("""
        ### 关于支气管炎风险检测系统
        
        该系统使用深度学习技术分析音频特征，评估支气管炎风险。
        
        #### 工作原理:
        1. 提取音频的MFCC(梅尔频率倒谱系数)特征
        2. 使用预训练的卷积神经网络模型进行分析
        3. 根据分析结果计算支气管炎风险概率
        
        #### 支持的音频格式:
        - WAV
        - MP3
        - M4A
        - FLAC
        
        #### 使用建议:
        - 录音时请保持环境安静
        - 录制清晰的呼吸声或语音
        - 录音时长建议5秒左右
        """)
        
        # 显示模型信息
        try:
            model, label_encoder = load_prediction_model()
            if model is not None:
                st.success("✅ 模型加载成功")
                st.write(f"可识别的音频类型: {list(label_encoder)}")
            else:
                st.error("❌ 模型加载失败")
        except Exception as e:
            st.error(f"❌ 无法加载模型信息: {str(e)}")

# 运行应用
if __name__ == "__main__":
    main()