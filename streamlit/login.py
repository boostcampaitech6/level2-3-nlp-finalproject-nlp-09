import hashlib
import pandas as pd

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def create_usertable(c):
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(c, db, username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	db.commit()

def login_user(c, username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def join_user(c, username):
  c.execute('SELECT * FROM userstable WHERE username = ?', (username,))
  data = c.fetchall()
  return data

def create_diarytable(c):
  c.execute('CREATE TABLE IF NOT EXISTS diarytable (diary_id TEXT, id INTEGER, date DATE, content TEXT, summary TEXT, emotion TEXT, word TEXT, PRIMARY KEY (diary_id, id))')

def load_user_data(c, username):
  c.execute('SELECT * FROM diarytable WHERE id = ?', (username,))
  data = c.fetchall()
  data = pd.DataFrame(data, columns=['diary_id', 'id', 'date', 'content', 'summary', 'emotion','word'])
  data = data.apply(lambda x:x.replace("'", '"'))
  return data