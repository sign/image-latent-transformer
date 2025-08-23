from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4')

texts = {
    "English": "Tours are cheaper for larger groups, so if you're by yourself or with just one friend, try to meet other people and form a group of four to six for a better per-person rate.", # noqa: E501
    "Italian": "I tour sono più economici per i gruppi più numerosi, quindi se sei da solo o con un solo amico, prova a incontrare altre persone e a formare un gruppo da quattro a sei persone per ottenere una tariffa più conveniente a persona.", # noqa: E501
    "German": "Touren sind für größere Gruppen günstiger. Wenn Sie also alleine oder mit nur einem Freund unterwegs sind, versuchen Sie, andere Leute kennenzulernen und eine Gruppe von vier bis sechs Personen zu bilden, um einen besseren Preis pro Person zu erhalten.", # noqa: E501
    "Finnish": "Retket ovat halvempia suuremmille ryhmille, joten jos olet yksin tai vain yhden ystävän kanssa, yritä tavata muita ihmisiä ja muodosta neljän tai kuuden hengen ryhmä saadaksesi paremman hinnan per henkilö.", # noqa: E501
    "Arabic": "تكون الجولات أرخص بالنسبة للمجموعات الكبيرة، لذلك إذا كنت بمفردك أو مع صديق واحد فقط، فحاول مقابلة أشخاص آخرين وتشكيل مجموعة مكونة من أربعة إلى ستة أشخاص للحصول على سعر أفضل للشخص الواحد.", # noqa: E501
    "Hebrew": "סיורים זולים יותר לקבוצות גדולות יותר, כך שאם אתם לבד או עם חבר אחד בלבד, נסו לפגוש אנשים אחרים וליצור קבוצה של ארבעה עד שישה אנשים לקבלת מחיר טוב יותר לאדם.", # noqa: E501
    "Shan": "ၶၢဝ်းတၢင်း တႃႇၸုမ်းယႂ်ႇၼၼ်ႉ ၵႃႈၶၼ်မၼ်း ထုၵ်ႇလိူဝ်လႄႈ သင်ဝႃႈ ၸဝ်ႈၵဝ်ႇ ယူႇႁင်းၵူၺ်း ဢမ်ႇၼၼ် မီးဢူၺ်းၵေႃႉ ၵေႃႉလဵဝ်ၵွႆးၼႆၸိုင် ၶတ်းၸႂ် ႁူပ်ႉထူပ်း ၵူၼ်းတၢင်ႇၵေႃႉသေ ႁဵတ်းၸုမ်း 4 ၵေႃႉ တေႃႇထိုင် 6 ၵေႃႉ ႁႂ်ႈလႆႈ ၵႃႈၶၼ် ၼိုင်ႈၵေႃႉ ဢၼ်လီလိူဝ်ၼၼ်ႉယဝ်ႉ။", # noqa: E501
}

print("| Language | Text (Google Translate) | Bytes (UTF-8) | Tokens (GPT-4) | Words (Whitespace) |")
print("|----------|-------------------------|-------------|----------------|--------------------|")
for lang, text in texts.items():
    num_bytes = len(text.encode('utf-8'))
    num_tokens = len(tokenizer(text)['input_ids'])
    num_words = len(text.split())
    print(f"| {lang} | {text} | {num_bytes} | {num_tokens} | {num_words} |")
