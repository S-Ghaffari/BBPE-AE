import hashlib
import requests

def check_pwned(passes):
  headers = {
    "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.8) Gecko/20100722 Firefox/3.6.8 GTB7.1 (.NET CLR 3.5.30729)", 
    "Referer": "https://haveibeenpwned.com/"
  }       #از این دیکشنری برای تنظیم هدر های بالا برای درخواست به ان ادرس استفاده میکنه--برای جلوگیری از مسدود شدن توسط سرور
  found_passwds = {}
  pwned_api = 'https://api.pwnedpasswords.com/range/'
  for p in passes:
    hash_object = hashlib.sha1(p.encode())    # -sha-1 رمزعبور را به عنوان یک عدد ده رقمی در مبنای 16 نشان میده
    pbHash = hash_object.hexdigest().upper()   #تبدیل به حروف بزرگ
    try:
      res = requests.get(pwned_api + pbHash[:5],  headers=headers, timeout=10)      #لیستی از پسوندهای هش راکه با پیشوند مطابقت دارند را بازیابی میکند
      range_hashes = res.text.split('\r\n')
      for h in range_hashes:
        h_c = h.split(':')          # hash : count
        if h_c[0] == pbHash[5:]:
          found_passwds[p] = h_c[1]
    except Exception as e:
      print(f'request timed out for pass {p}')
  return found_passwds


with open("Generated Passwords-10epoch-hotmail-1000.txt", "rt", encoding='UTF-8', errors='ignore') as file:
        gen_pass = file.read().split()
#print(gen_pass)
fnd_pass = check_pwned(gen_pass)
#print(fnd_pass)
with open("check_pwned-10epoch-hotmail-1000.txt", "w") as file:
    for password, count in fnd_pass.items():
        file.write(f"{password}: {count}\n")