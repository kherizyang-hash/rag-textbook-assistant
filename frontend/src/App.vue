<template>
  <div class="app">
    <div class="header">
      <h1>教材智能助手</h1>
      <p>基于《人工智能引论》RAG 问答系统</p>
    </div>
    <div class="chat-container">
      <div class="messages" ref="messagesContainer">
        <div v-for="msg in messages" :key="msg.id" :class="['message', msg.role]">
          <div class="content" v-html="marked(msg.content)"></div>
        </div>
        <div v-if="isLoading" class="message assistant">
            <div class="content thinking">
                <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
        </div>
      </div>
      <div class="input-area">
        <input
          type="text"
          v-model="question"
          @keyup.enter="sendMessage"
          placeholder="输入问题，按 Enter 发送"
        />
        <button @click="sendMessage">发送</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { marked } from 'marked'
import { ref, nextTick } from 'vue'

const messagesContainer = ref(null)  // 消息容器的引用

const messages = ref([
  { id: 1, role: 'assistant', content: '你好！我是你的教材助教。请问有什么关于《人工智能引论》的问题？' }
])
const question = ref('')
const isLoading = ref(false)

async function sendMessage() {
  if (!question.value.trim() || isLoading.value) return

  const userQuestion = question.value
  messages.value.push({ id: Date.now(), role: 'user', content: userQuestion })
  question.value = ''
  scrollToBottom()

  isLoading.value = true
  scrollToBottom()  // 显示加载动画时也滚动

  try {
    const res = await fetch('http://localhost:8000/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: userQuestion })
    })
    const data = await res.json()
    messages.value.push({ id: Date.now(), role: 'assistant', content: data.answer })
    scrollToBottom()
  } catch (err) {
    messages.value.push({ id: Date.now(), role: 'assistant', content: '连接后端失败：' + err.message })
    scrollToBottom()
  } finally {
    isLoading.value = false
  }
}
function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}
</script>

<style scoped>
.app {
  max-width: 800px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f5f5f5;
}
.header {
  text-align: center;
  padding: 20px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}
.header h1 { margin: 0; font-size: 24px; }
.header p { margin: 5px 0 0; font-size: 12px; opacity: 0.9; }
.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 15px;
}
.message {
  display: flex;
  margin-bottom: 15px;
}
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }
.message .content {
  max-width: 70%;
  padding: 10px 15px;
  border-radius: 18px;
  background: white;
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.message.user .content { background: #667eea; color: white; }
.message.assistant .content { background: white; }
.thinking { color: #999; font-style: italic; }
.input-area {
  display: flex;
  padding: 15px;
  background: white;
  border-top: 1px solid #ddd;
  gap: 10px;
}
.input-area input {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 25px;
  outline: none;
}
.input-area button {
  padding: 12px 24px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 25px;
  cursor: pointer;
}
.thinking .dot {
  display: inline-block;
  animation: wave 1s infinite;
}
.thinking .dot:nth-child(2) { animation-delay: 0.2s; }
.thinking .dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes wave {
  0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
  30% { opacity: 1; transform: translateY(-5px); }
}
</style>