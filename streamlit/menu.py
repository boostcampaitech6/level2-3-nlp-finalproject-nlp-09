import streamlit as st


def authenticated_menu():
    # Show a navigation menu for authenticated users
    if st.sidebar.button('로그아웃'):
        st.session_state['is_login'] = False
        st.rerun()
    st.sidebar.page_link("pages/diary.py", label="오늘의 일기")
    st.sidebar.page_link("pages/summary.py", label="요약")
    st.sidebar.page_link("pages/mypage.py", label="마이페이지")


def unauthenticated_menu():
    # Show a navigation menu for unauthenticated users
    st.sidebar.page_link("main.py", label="로그인")


def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    if "is_login" not in st.session_state or st.session_state['is_login'] == False:
        return
    authenticated_menu()


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    if "is_login" not in st.session_state or st.session_state['is_login'] == False:
        st.switch_page("main.py")
    menu()