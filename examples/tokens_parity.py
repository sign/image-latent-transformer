import matplotlib.pyplot as plt
import pandas as pd
from transformers import GPT2TokenizerFast

from image_latent_transformer.pretokenizer.pretokenizer import text_to_words

tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4')

# Warm chinese and japanese models, because it prints stuff
text_to_words("体")

texts = {
    "English": "Tours are cheaper for larger groups, so if you're by yourself or with just one friend, try to meet other people and form a group of four to six for a better per-person rate.",
    "Italian": "I tour sono più economici per i gruppi più numerosi, quindi se sei da solo o con un solo amico, prova a incontrare altre persone e a formare un gruppo da quattro a sei persone per ottenere una tariffa più conveniente a persona.",
    "German": "Touren sind für größere Gruppen günstiger. Wenn Sie also alleine oder mit nur einem Freund unterwegs sind, versuchen Sie, andere Leute kennenzulernen und eine Gruppe von vier bis sechs Personen zu bilden, um einen besseren Preis pro Person zu erhalten.",
    "Chinese": "团体旅游价格更便宜，所以如果您独自一人或只有一个朋友，请尝试结识其他人并组成一个四到六人的团体，以获得更好的每人价格。",
    "Japanese": "ツアーはグループが多ければ安くなるので、一人または友達とだけ参加する場合は、他の人と会って4人から6人のグループを作ると、一人当たりの料金が安くなります。",
    "Finnish": "Retket ovat halvempia suuremmille ryhmille, joten jos olet yksin tai vain yhden ystävän kanssa, yritä tavata muita ihmisiä ja muodosta neljän tai kuuden hengen ryhmä saadaksesi paremman hinnan per henkilö.",
    "Russian": "Туры обходятся дешевле для больших групп, поэтому, если вы одни или с одним другом, постарайтесь познакомиться с другими людьми и сформировать группу из четырех-шести человек, чтобы получить более выгодную цену на человека.",
    "Arabic": "تكون الجولات أرخص بالنسبة للمجموعات الكبيرة، لذلك إذا كنت بمفردك أو مع صديق واحد فقط، فحاول مقابلة أشخاص آخرين وتشكيل مجموعة مكونة من أربعة إلى ستة أشخاص للحصول على سعر أفضل للشخص الواحد.",
    "Hebrew": "סיורים זולים יותר לקבוצות גדולות יותר, כך שאם אתם לבד או עם חבר אחד בלבד, נסו לפגוש אנשים אחרים וליצור קבוצה של ארבעה עד שישה אנשים לקבלת מחיר טוב יותר לאדם.",
    "Greek": "Οι εκδρομές είναι φθηνότερες για μεγαλύτερες ομάδες, οπότε αν είστε μόνοι σας ή με έναν μόνο φίλο, προσπαθήστε να γνωρίσετε άλλα άτομα και να σχηματίσετε μια ομάδα τεσσάρων έως έξι ατόμων για καλύτερη τιμή ανά άτομο.",
    "Tamil": "பெரிய குழுக்களுக்கு சுற்றுலாக்கள் மலிவானவை, எனவே நீங்கள் தனியாகவோ அல்லது ஒரு நண்பருடனோ இருந்தால், மற்றவர்களைச் சந்தித்து நான்கு முதல் ஆறு பேர் கொண்ட குழுவை உருவாக்கி, ஒரு நபருக்கு சிறந்த விலையைப் பெற முயற்சிக்கவும்.",
    "Kannada": "ದೊಡ್ಡ ಗುಂಪುಗಳಿಗೆ ಪ್ರವಾಸಗಳು ಅಗ್ಗವಾಗಿರುತ್ತವೆ, ಆದ್ದರಿಂದ ನೀವು ಒಬ್ಬಂಟಿಯಾಗಿ ಅಥವಾ ಒಬ್ಬ ಸ್ನೇಹಿತನೊಂದಿಗೆ ಇದ್ದರೆ, ಇತರ ಜನರನ್ನು ಭೇಟಿ ಮಾಡಲು ಪ್ರಯತ್ನಿಸಿ ಮತ್ತು ಪ್ರತಿ ವ್ಯಕ್ತಿಗೆ ಉತ್ತಮ ದರಕ್ಕಾಗಿ ನಾಲ್ಕರಿಂದ ಆರು ಜನರ ಗುಂಪನ್ನು ರಚಿಸಿ.",
    "Shan": "ၶၢဝ်းတၢင်း တႃႇၸုမ်းယႂ်ႇၼၼ်ႉ ၵႃႈၶၼ်မၼ်း ထုၵ်ႇလိူဝ်လႄႈ သင်ဝႃႈ ၸဝ်ႈၵဝ်ႇ ယူႇႁင်းၵူၺ်း ဢမ်ႇၼၼ် မီးဢူၺ်းၵေႃႉ ၵေႃႉလဵဝ်ၵွႆးၼႆၸိုင် ၶတ်းၸႂ် ႁူပ်ႉထူပ်း ၵူၼ်းတၢင်ႇၵေႃႉသေ ႁဵတ်းၸုမ်း 4 ၵေႃႉ တေႃႇထိုင် 6 ၵေႃႉ ႁႂ်ႈလႆႈ ၵႃႈၶၼ် ၼိုင်ႈၵေႃႉ ဢၼ်လီလိူဝ်ၼၼ်ႉယဝ်ႉ။",
}

data = {
    "Language": [],
    "Bytes (UTF-8)": [],
    "Tokens (GPT-4)": [],
    "Words (Whitespace+)": [],
}

print("| Language | Text (Google Translate) | Bytes (UTF-8) | Tokens (GPT-4) | Words (Whitespace+) |")
print("|----------|-------------------------|-------------|----------------|--------------------|")
for lang, text in texts.items():
    data["Language"].append(lang)
    num_bytes = len(text.encode('utf-8'))
    data["Bytes (UTF-8)"].append(num_bytes)
    num_tokens = len(tokenizer(text)['input_ids'])
    data["Tokens (GPT-4)"].append(num_tokens)
    num_words = len(text_to_words(text))
    data["Words (Whitespace+)"].append(num_words)
    print(f"| {lang} | {text} | {num_bytes} | {num_tokens} | {num_words} |")

df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.set_index("Language").plot(kind="bar", figsize=(12, 7))

plt.title("Text Size Across Languages (Bytes, Tokens, Words)")
plt.ylabel("Count")
plt.xlabel("Language")
plt.xticks(rotation=45)
plt.legend(title="Measure")
plt.tight_layout()
plt.savefig("../assets/parity/tokenization-parity-words.png")
