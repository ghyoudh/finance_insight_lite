"""
All custom CSS for Finance Insight Lite.
Injected via st.html() to avoid the st.markdown style-rendering bug.
"""

def get_css(lang: str = "en") -> str:
    direction = "rtl" if lang == "ar" else "ltr"
    text_align = "right" if lang == "ar" else "left"

    heading_font = "'Cairo', sans-serif" if lang == "ar" else "'Space Grotesk', sans-serif"
    body_font = "'Tajawal', sans-serif" if lang == "ar" else "'Outfit', sans-serif"

    lang_btn_side = "right" if lang == "en" else "left"
    browse_text = "إضافة ملفات" if lang == "ar" else "Add Files"

    # جهة كل رسالة: انجليزي = يوزر يسار / بوت يمين || عربي = يوزر يمين / بوت يسار
    if lang == "ar":
        user_side, bot_side = "right", "left"
        user_flex, bot_flex = "row-reverse", "row"
    else:
        user_side, bot_side = "left", "right"
        user_flex, bot_flex = "row", "row-reverse"

    def _margins(side):
        return ("auto", "0") if side == "right" else ("0", "auto")

    def _radius(side):
        # حواف دائرية أكثر مع ذيل ناعم بالزاوية القريبة من الأفتار
        return "26px 26px 10px 26px" if side == "right" else "26px 26px 26px 10px"

    user_margin_l, user_margin_r = _margins(user_side)
    bot_margin_l, bot_margin_r = _margins(bot_side)
    user_radius = _radius(user_side)
    bot_radius = _radius(bot_side)

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Outfit:wght@300;400;500;600&family=Tajawal:wght@400;500;700;800&family=Space+Grotesk:wght@400;500;600;700&family=Cairo:wght@400;600;700;800;900&display=swap');

:root {{
    --navy-900: #060F1E;
    --navy-800: #0A1628;
    --navy-700: #0F1F38;
    --navy-600: #132039;
    --navy-500: #1B2A4A;
    --navy-400: #243656;
    --accent:   #3B82F6;
    --accent-light: #60A5FA;
    --accent-glow: rgba(59, 130, 246, 0.45);
    --accent-glow-strong: rgba(59, 130, 246, 0.7);
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --success: #22C55E;
    --warning: #F59E0B;
    --error:   #EF4444;
    --card-bg: #0F1D32;
    --card-border: #1E3050;
    --radius: 12px;
    --glow-transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --font-heading: {heading_font};
    --font-body: {body_font};
    --navy-chat-user-1: #24365E;
    --navy-chat-user-2: #16223E;
    --navy-chat-bot-1:  #2C4A8A;
    --navy-chat-bot-2:  #16264A;
}}

.stApp, .main .block-container {{
    direction: {direction};
    text-align: {text_align};
}}
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
    font-family: var(--font-body);
}}

#MainMenu, footer, header {{
    visibility: hidden;
}}

[data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    width: 0 !important;
    height: 0 !important;
    line-height: 0 !important;
    overflow: hidden !important;
    display: inline-block !important;
}}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--navy-800); }}
::-webkit-scrollbar-thumb {{ background: var(--navy-400); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

div.stButton > button {{
    background: linear-gradient(135deg, var(--navy-500), var(--navy-400));
    color: var(--text-primary);
    border: 1px solid var(--card-border);
    border-radius: 50px;
    padding: 0.55rem 1.6rem;
    font-family: var(--font-heading);
    font-weight: 600;
    transition: var(--glow-transition);
    width: 100%;
}}
div.stButton > button:hover {{
    background: linear-gradient(135deg, var(--accent), #2563EB);
    border-color: var(--accent);
    box-shadow: 0 0 18px var(--accent-glow), 0 0 40px rgba(59,130,246,0.18);
    color: #fff;
    transform: translateY(-1px);
}}

div[data-testid="stHorizontalBlock"] div.stButton > button {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.75rem 1.1rem;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 0.6rem;
    text-align: {text_align};
}}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {{
    background: var(--navy-500);
    border-color: var(--accent);
    box-shadow: 0 0 14px var(--accent-glow);
    transform: translateY(-2px);
}}

/* ═══════════ هالة ضوء ناعمة تلف اللوقو من كل الجهات ═══════════ */
.logo-halo-wrap {{
    position: relative;
    display: inline-block;
    z-index: 1;
}}
.logo-halo-wrap::before {{
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 145%;
    height: 145%;
    transform: translate(-50%, -50%);
    border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.32) 0%, rgba(59,130,246,0.14) 45%, transparent 72%);
    filter: blur(16px);
    z-index: -1;
    animation: haloPulse 5s ease-in-out infinite;
    pointer-events: none;
}}
@keyframes haloPulse {{
    0%, 100% {{ opacity: 0.6; }}
    50%      {{ opacity: 1; }}
}}
@media (prefers-reduced-motion: reduce) {{
    .logo-halo-wrap::before {{ animation: none; opacity: 0.85; }}
}}

div[data-testid="stFileUploader"] label {{
    text-align: center !important;
    display: flex;
    justify-content: center;
    width: 100%;
}}
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"] {{
    display: flex !important;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    text-align: center;
    padding: 1.6rem 1.2rem;
    outline: 2px dashed var(--navy-400);
    outline-offset: -2px;
    border-radius: var(--radius);
    background: rgba(15, 29, 50, 0.35);
    transition: var(--glow-transition);
    width: 100%;
}}
[data-testid="stFileUploaderDropzone"] button span {{
    display: none !important;
}}
[data-testid="stFileUploaderDropzone"] button::after {{
    content: "{browse_text}";
    display: block;
}}
section[data-testid="stFileUploaderDropzone"]::before,
div[data-testid="stFileUploaderDropzone"]::before {{
    content: "";
    display: block;
    width: 38px;
    height: 38px;
    margin: 0 auto 0.4rem;
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    background-image: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2360A5FA%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M7%2018a4.5%204.5%200%2001-1.2-8.84A5.5%205.5%200%200116.5%207.5%204%204%200%200117%2015.9%22/%3E%3Cpath%20d%3D%22M12%2011v8%22/%3E%3Cpath%20d%3D%22M9%2014l3-3%203%203%22/%3E%3C/svg%3E");
}}

div[data-testid="stElementContainer"]:has(> .upload-cta-label) {{
    margin-bottom: -0.25rem !important;
}}
.upload-cta-label {{
    font-family: var(--font-heading);
    font-weight: 500;
    letter-spacing: 0.01em;
    transition: var(--glow-transition);
}}
div[data-testid="stElementContainer"]:has(> .upload-cta-label):hover ~ div[data-testid="stElementContainer"] section[data-testid="stFileUploaderDropzone"],
div[data-testid="stElementContainer"]:has(> .upload-cta-label) + div[data-testid="stElementContainer"]:hover section[data-testid="stFileUploaderDropzone"],
div[data-testid="stElementContainer"]:has(> .upload-cta-label):hover ~ div[data-testid="stElementContainer"] div[data-testid="stFileUploaderDropzone"],
div[data-testid="stElementContainer"]:has(> .upload-cta-label) + div[data-testid="stElementContainer"]:hover div[data-testid="stFileUploaderDropzone"] {{
    outline-color: var(--accent) !important;
    box-shadow: 0 0 22px var(--accent-glow);
    background: rgba(59,130,246,0.07);
}}
div[data-testid="stElementContainer"]:has(> .upload-cta-label):hover {{
    color: var(--accent-light);
}}

div[data-testid="stFileUploaderFile"] {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    margin-top: 0.4rem;
}}
div[data-testid="stFileUploaderFile"] button {{
    background: rgba(239, 68, 68, 0.12) !important;
    border: 1px solid rgba(239, 68, 68, 0.45) !important;
    border-radius: 50% !important;
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    position: relative;
    transition: var(--glow-transition);
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    overflow: hidden;
}}
div[data-testid="stFileUploaderFile"] button:hover {{
    background: rgba(239, 68, 68, 0.28) !important;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
}}
div[data-testid="stFileUploaderFile"] button [data-testid="stIconMaterial"] {{
    display: none !important;
}}
div[data-testid="stFileUploaderFile"] button span,
div[data-testid="stFileUploaderFile"] button p,
div[data-testid="stFileUploaderFile"] button div {{
    display: none !important;
}}
div[data-testid="stFileUploaderFile"] button::after {{
    content: "×" !important;
    color: var(--error) !important;
    font-size: 1.05rem !important;
    font-weight: 800;
    line-height: 1;
}}
/* أي زر إضافي (مثل "Add Files" ثانية) يظهر مباشرة بجانب صف الملف يتم إخفاؤه،
   عشان ما يبقى إلا زر الـ × جنب اسم الملف. زر "Add Files" الرئيسي فوق الدروب زون يبقى ظاهر عادي. */
div[data-testid="stFileUploaderFile"] ~ button {{
    display: none !important;
}}

/* ═══════════════════════════════════════════════════════════
   محادثة الشات — تصميم فقاعات (WhatsApp style)
   كل فقاعة بعرض محتواها الفعلي (fit-content) وتتمحور يمين/يسار
   حسب الدور واللغة عبر margin auto على عنصر block-level حقيقي
   (بدل inline-flex اللي كان يعطّل الـ margin auto سابقًا).
   ═══════════════════════════════════════════════════════════ */
.msg-user-anchor, .msg-bot-anchor {{ display: none; }}

div[data-testid="stElementContainer"]:has(> .msg-user-anchor),
div[data-testid="stElementContainer"]:has(> .msg-bot-anchor) {{
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible;
    border: none !important;
}}

/* الفقاعة الأساسية */
div[data-testid="stChatMessage"] {{
    display: flex !important;
    direction: ltr !important; /* نتحكم بالجهة يدويًا، منفصل عن اتجاه الصفحة */
    align-items: flex-end;
    gap: 0.55rem;
    width: fit-content;
    max-width: 68%;
    border-radius: 26px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.85rem;
    position: relative;
}}
@media (max-width: 720px) {{
    div[data-testid="stChatMessage"] {{ max-width: 86%; }}
}}
div[data-testid="stChatMessage"] img {{
    width: 34px;
    height: 34px;
    flex-shrink: 0;
    filter: brightness(0.55) contrast(1.15) saturate(1.1);
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {{
    text-align: {text_align} !important;
    direction: {direction} !important;
}}

/* رسالة اليوزر — نستهدفها مباشرة عبر aria-label، مضمونة بغض النظر عن ترتيب DOM */
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {{
    flex-direction: {user_flex} !important;
    margin-left: {user_margin_l} !important;
    margin-right: {user_margin_r} !important;
    border-radius: {user_radius} !important;
    background: linear-gradient(135deg, var(--navy-chat-user-1), var(--navy-chat-user-2)) !important;
    border: 1px solid var(--card-border) !important;
}}

/* رسالة البوت — إضاءة على كامل حدود الفقاعة بدل الشعاع الطولي */
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {{
    flex-direction: {bot_flex} !important;
    margin-left: {bot_margin_l} !important;
    margin-right: {bot_margin_r} !important;
    border-radius: {bot_radius} !important;
    background: linear-gradient(145deg, var(--navy-chat-bot-1), var(--navy-chat-bot-2)) !important;
    border: 1px solid rgba(96, 165, 250, 0.45) !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25), 0 0 10px var(--accent-glow) !important;
    animation: chatBorderPulse 2.6s ease-in-out infinite;
}}
@keyframes chatBorderPulse {{
    0%, 100%  {{ border-color: rgba(96, 165, 250, 0.35); box-shadow: 0 4px 18px rgba(0,0,0,0.25), 0 0 8px var(--accent-glow); }}
    50%       {{ border-color: rgba(96, 165, 250, 0.7);  box-shadow: 0 4px 18px rgba(0,0,0,0.25), 0 0 16px var(--accent-glow-strong); }}
}}
@media (prefers-reduced-motion: reduce) {{
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {{ animation: none; border-color: rgba(96,165,250,0.55); }}
}}

div[data-testid="stChatInput"] {{
    border-color: var(--card-border) !important;
    border-radius: 16px !important;
    background: var(--card-bg) !important;
    transition: var(--glow-transition);
}}
div[data-testid="stChatInput"]:focus-within {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 14px var(--accent-glow) !important;
}}
div[data-testid="stChatInput"] textarea {{
    color: var(--text-primary) !important;
    direction: {direction};
}}

hr {{
    border: none !important;
    border-top: 1px solid var(--card-border) !important;
    height: 0 !important;
    margin: 0.9rem 0 !important;
}}

.stApp {{ min-height: 100vh; }}
.main .block-container:has(.welcome-container) {{
    position: relative;
    top: 50%;
    transform: translateY(-50%);
    padding-top: 1rem;
    padding-bottom: 1rem;
}}

.main .block-container:not(:has(.welcome-container)) {{
    padding-top: 0.2rem !important;
}}

.welcome-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 1rem;
    direction: {direction};
}}
.welcome-logo {{ width: 190px; height: 190px; border-radius: 50%; margin-bottom: 0.2rem; position: relative; z-index: 1; }}

.welcome-title {{
    font-family: var(--font-heading);
    font-size: 3.3rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    background: linear-gradient(135deg, #DBEAFE 0%, var(--accent-light) 45%, var(--accent) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(0 0 16px var(--accent-glow));
    margin-bottom: 0.35rem;
}}
.welcome-subtitle {{
    font-family: var(--font-heading);
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--accent-light);
    letter-spacing: 0.01em;
    margin-bottom: 1.3rem;
}}
.welcome-desc {{
    font-size: 1rem;
    color: var(--text-muted);
    max-width: 540px;
    line-height: 1.65;
    margin-bottom: 1.6rem;
}}

.lang-btn-anchor {{ display: none; }}
div[data-testid="stElementContainer"]:has(> .lang-btn-anchor) {{ height: 0 !important; margin: 0 !important; padding: 0 !important; overflow: visible; border: none !important; }}
div[data-testid="stElementContainer"]:has(> .lang-btn-anchor) + div[data-testid="stElementContainer"] {{
    position: fixed;
    top: 26px;
    {lang_btn_side}: 28px !important;
    z-index: 999999;
    width: auto;
    margin: 0 !important;
}}

.dash-header-block {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding-top: 0.4rem;
    margin-bottom: 0.6rem;
}}
.dash-logo-row {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    direction: ltr !important;
}}
.dash-logo-row img {{ width: 80px; height: 80px; border-radius: 50%; }}
.dash-title {{
    font-family: var(--font-heading);
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.1;
    background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-light) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}}
.dash-title .accent {{
    background: linear-gradient(135deg, var(--accent-light), var(--accent));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(0 0 10px var(--accent-glow));
}}

.st-key-header_clear_chat_wrap {{
    display: flex;
    justify-content: flex-end;
    direction: ltr !important;
}}
.st-key-header_clear_chat_wrap div.stButton > button {{
    width: auto;
    background: transparent;
    border: 1px solid var(--card-border);
    border-radius: 999px !important;
    padding: 0.45rem 1.4rem !important;
    font-size: 0.82rem;
    font-weight: 500;
}}
.st-key-header_clear_chat_wrap div.stButton > button:hover {{
    box-shadow: 0 0 14px var(--accent-glow);
}}

/* زر الرجوع — ثابت فوق يسار دائمًا بغض النظر عن اللغة */
.back-btn-anchor {{ display: none; }}
div[data-testid="stElementContainer"]:has(> .back-btn-anchor) {{ height: 0 !important; margin: 0 !important; padding: 0 !important; overflow: visible; }}
div[data-testid="stElementContainer"]:has(> .back-btn-anchor) + div[data-testid="stElementContainer"] {{
    direction: ltr !important;
    position: fixed !important;
    inset-inline-start: 28px !important;
    inset-inline-end: auto !important;
    top: 24px !important;
    left: 28px !important;
    right: auto !important;
    z-index: 999999;
    width: auto;
    margin: 0 !important;
}}
div[data-testid="stElementContainer"]:has(> .back-btn-anchor) + div[data-testid="stElementContainer"] div.stButton > button {{
    width: 44px;
    height: 44px;
    padding: 0;
    border-radius: 50%;
    font-size: 1.15rem;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: fixed !important;
    top: 24px !important;
    left: 28px !important;
    right: auto !important;
}}

.features-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    max-width: 700px;
    margin: 0.4rem auto 0;
}}
.feature-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1rem 0.8rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 0.3rem;
    transition: var(--glow-transition);
    cursor: default;
}}
.feature-card:hover {{
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-glow), inset 0 0 12px rgba(59,130,246,0.08);
    transform: translateY(-3px);
}}
.feature-icon {{ margin-bottom: 0.15rem; }}
.feature-icon img {{ width: 24px; height: 24px; }}
.feature-title {{
    font-family: var(--font-heading);
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--text-primary);
}}
.feature-desc {{
    font-size: 0.76rem;
    color: var(--text-muted);
    line-height: 1.4;
}}
@media (max-width: 640px) {{
    .features-grid {{ grid-template-columns: 1fr; }}
}}

.ask-header {{ display: flex; align-items: center; gap: 0.55rem; margin: 0.4rem 0 0.15rem; direction: {direction}; }}
.ask-title {{ font-family: var(--font-heading); font-weight: 700; font-size: 1.35rem; margin: 0; }}
.ask-subtitle {{ color: var(--text-muted); font-size: 0.92rem; margin: 0 0 1.3rem; direction: {direction}; }}
</style>
"""