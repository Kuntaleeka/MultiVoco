"""Fixed lines the agent says without asking the model.

The Kannada and Bengali lines need a native speaker's review before M2k and M5.
None of them may contain a figure: they are spoken when a figure could not be trusted.
"""

from app.core.languages import Lang

# Used while the session language is still being detected.
NEUTRAL_GREETING = "Hello. Namaste. Namaskara. Nomoshkar."

GREETINGS: dict[Lang, str] = {
    Lang.EN: "Hello, this is the loan helpline. How can I help you today?",
    Lang.HI: "नमस्ते, यह लोन हेल्पलाइन है। मैं आपकी क्या मदद कर सकती हूँ?",
    Lang.KN: "ನಮಸ್ಕಾರ, ಇದು ಲೋನ್ ಸಹಾಯವಾಣಿ. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
    Lang.BN: "নমস্কার, এটি লোন হেল্পলাইন। আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
}

# Said when a reply failed a guardrail twice, or nothing was heard.
SAFE_REPLY: dict[Lang, str] = {
    Lang.EN: "Sorry, I could not confirm that. Could you say that once more?",
    Lang.HI: "माफ़ कीजिए, मैं यह पक्का नहीं कर पाई। क्या आप एक बार फिर बताएँगे?",
    Lang.KN: "ಕ್ಷಮಿಸಿ, ಅದನ್ನು ನಾನು ಖಚಿತಪಡಿಸಲು ಆಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಇನ್ನೊಮ್ಮೆ ಹೇಳುತ್ತೀರಾ?",
    Lang.BN: "দুঃখিত, আমি এটা নিশ্চিত করতে পারলাম না। আপনি কি আর একবার বলবেন?",
}

# Said when the code, not the model, decides to hand the call over.
HANDOFF_LINE: dict[Lang, str] = {
    Lang.EN: "I am connecting you to a colleague who can help. Please stay on the line.",
    Lang.HI: "मैं आपको हमारे एक सहयोगी से जोड़ रही हूँ। कृपया लाइन पर बने रहें।",
    Lang.KN: "ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ನಮ್ಮ ಸಹೋದ್ಯೋಗಿಗೆ ಕರೆಯನ್ನು ವರ್ಗಾಯಿಸುತ್ತಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ಲೈನ್‌ನಲ್ಲೇ ಇರಿ.",
    Lang.BN: "আমি আপনাকে আমাদের একজন সহকর্মীর সঙ্গে যুক্ত করছি। অনুগ্রহ করে লাইনে থাকুন।",
}
