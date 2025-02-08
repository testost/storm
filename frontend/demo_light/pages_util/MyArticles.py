import os

import demo_util
import streamlit as st
from demo_util import DemoFileIOHelper, DemoUIHelper


# set page config and display title
def my_articles_page():
    # Add custom CSS
    st.markdown("""
        <style>
        /* Make delete button smaller */
        button[data-testid="baseButton-secondary"]:has(div:contains("🗑️")) {
            padding: 0px 8px;
            height: 24px;
            line-height: 24px;
            margin-top: 4px;
        }
        /* Adjust caption size and spacing */
        .st-emotion-cache-q8sbsg {
            font-size: 0.8em;
            color: #666;
            margin-top: 8px;
        }
        /* Make article title buttons look like text */
        button[data-testid="baseButton-secondary"]:not(:has(div:contains("🗑️"))) {
            text-align: left;
            background: none;
            border: none;
            padding: 4px 0;
            margin: 0;
            font-size: 1em;
            color: #000;
        }
        button[data-testid="baseButton-secondary"]:not(:has(div:contains("🗑️"))):hover {
            color: #00A0DC;
            background: none;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        _, return_button_col = st.columns([2, 5])
        with return_button_col:
            if st.button(
                "Select another article",
                disabled="page2_selected_my_article" not in st.session_state,
            ):
                if "page2_selected_my_article" in st.session_state:
                    del st.session_state["page2_selected_my_article"]
                st.rerun()

    # sync my articles
    if "page2_user_articles_file_path_dict" not in st.session_state:
        local_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
        os.makedirs(local_dir, exist_ok=True)
        st.session_state["page2_user_articles_file_path_dict"] = (
            DemoFileIOHelper.read_structure_to_dict(local_dir)
        )

    # if no feature demo selected, display all featured articles as info cards
    def display_article_item(article_name):
        cleaned_article_title = article_name.replace("_", " ")
        article_path = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR", article_name)
        creation_time = DemoFileIOHelper.get_latest_modification_time(article_path)
        
        col1, col2, col3 = st.columns([6, 3, 1])
        with col1:
            if st.button(cleaned_article_title, key=f"select_{article_name}"):
                st.session_state["page2_selected_my_article"] = article_name
                st.rerun()
        with col2:
            st.caption(creation_time)
        with col3:
            if st.button("🗑️", key=f"delete_{article_name}", help="Delete article", use_container_width=False):
                import shutil
                shutil.rmtree(article_path)
                if "page2_user_articles_file_path_dict" in st.session_state:
                    del st.session_state["page2_user_articles_file_path_dict"]
                st.rerun()

    if "page2_selected_my_article" not in st.session_state:
        # display articles as a list
        if len(st.session_state["page2_user_articles_file_path_dict"]) > 0:
            st.markdown("### Your Articles")
            
            # Get articles with their creation times
            articles_with_times = []
            for article_name in st.session_state["page2_user_articles_file_path_dict"].keys():
                article_path = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR", article_name)
                creation_time = DemoFileIOHelper.get_latest_modification_time(article_path)
                articles_with_times.append((article_name, creation_time))
            
            # Sort by creation time, newest first
            articles_with_times.sort(key=lambda x: x[1], reverse=True)
            
            # Display sorted articles
            for article_name, _ in articles_with_times:
                display_article_item(article_name)
        else:
            st.info("No articles yet. Create your first article!")
    else:
        selected_article_name = st.session_state["page2_selected_my_article"]
        selected_article_file_path_dict = st.session_state[
            "page2_user_articles_file_path_dict"
        ][selected_article_name]

        demo_util.display_article_page(
            selected_article_name=selected_article_name,
            selected_article_file_path_dict=selected_article_file_path_dict,
            show_title=True,
            show_main_article=True,
        )
