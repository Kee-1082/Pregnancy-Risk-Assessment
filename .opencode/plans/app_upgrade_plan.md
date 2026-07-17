# App UI Enhancement + Floating Chatbot Plan

## Step 1: Replace CSS (lines 29-366) with enhanced version
- Refined gold palette (#D4AF37 accent, #FFFAF5 bg)
- Glassmorphism effects on cards/expanders
- Slide-up animations on risk results
- Better sidebar nav styling

## Step 2: Add floating_chat_widget() function
- Floating gold chat bubble (bottom-right, fixed)
- Expandable panel with FAQ chips + Gemini AI chat
- Reuses existing get_ai_client()

## Step 3: Update main() to call floating_chat_widget()
- One line addition at end of main()

## Step 4: Update sidebar navigation text
- Polish sidebar header
