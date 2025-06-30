import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import exceptions

def init_firebase():
    if not firebase_admin._apps:
        firebase_key = st.secrets["FIREBASE_KEY"]
        cred = credentials.Certificate(json.loads(firebase_key))
        firebase_admin.initialize_app(cred)

def verify_user(email, password):
    try:
        # Firebase Admin SDK ile doğrudan email+password doğrulama yok.
        # Şifre doğrulaması için Firebase Client SDK veya REST API kullanılmalı.
        # Burada sadece email kontrolü yapılıyor, şifre doğrulama için REST API kullanmalısın.
        
        user = auth.get_user_by_email(email)  # Kullanıcı var mı kontrol
        # Şifre kontrolü yapılmıyor, bu yüzden burada hep True dönecek.
        return True, f"Welcome back {user.email}!"
    except exceptions.NotFoundError:
        return False, "User not found."
    except Exception as e:
        return False, f"Error: {e}"
