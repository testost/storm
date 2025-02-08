import os
import time
import re

import demo_util
import streamlit as st
from demo_util import (
    DemoFileIOHelper,
    DemoTextProcessingHelper,
    DemoUIHelper,
    truncate_filename,
)


def handle_not_started():
    if st.session_state["page3_write_article_state"] == "not started":
        _, search_form_column, _ = st.columns([2, 5, 2])
        with search_form_column:
            with st.form(key="search_form"):
                # Text input for the search topic
                DemoUIHelper.st_markdown_adjust_size(
                    content="Enter the topic you want to learn in depth:", font_size=18
                )
                st.session_state["page3_topic"] = st.text_area(
                    label="page3_topic",
                    label_visibility="collapsed",
                    height=100,
                    max_chars=1000,  # Set a high limit that's well within LLM context lengths
                    help="Describe the topic you want to research. You can provide additional context or specific aspects you're interested in."
                )
                pass_appropriateness_check = True

                # Submit button for the form
                submit_button = st.form_submit_button(label="Research")
                # only start new search when button is clicked, not started, or already finished previous one
                if submit_button and st.session_state["page3_write_article_state"] in [
                    "not started",
                    "show results",
                ]:
                    if not st.session_state["page3_topic"].strip():
                        pass_appropriateness_check = False
                        st.session_state["page3_warning_message"] = (
                            "topic could not be empty"
                        )

                    st.session_state["page3_topic_name_cleaned"] = (
                        st.session_state["page3_topic"]
                        .replace(" ", "_")
                        .replace("/", "_")
                    )
                    st.session_state["page3_topic_name_truncated"] = truncate_filename(
                        st.session_state["page3_topic_name_cleaned"]
                    )
                    if not pass_appropriateness_check:
                        st.session_state["page3_write_article_state"] = "not started"
                        alert = st.warning(
                            st.session_state["page3_warning_message"], icon="⚠️"
                        )
                        time.sleep(5)
                        alert.empty()
                    else:
                        st.session_state["page3_write_article_state"] = "initiated"


def handle_initiated():
    if st.session_state["page3_write_article_state"] == "initiated":
        current_working_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
        if not os.path.exists(current_working_dir):
            os.makedirs(current_working_dir)

        if "runner" not in st.session_state:
            demo_util.set_storm_runner()
        st.session_state["page3_current_working_dir"] = current_working_dir
        st.session_state["page3_write_article_state"] = "pre_writing"


def handle_pre_writing():
    if st.session_state["page3_write_article_state"] == "pre_writing":
        status = st.status(
            "I am brain**STORM**ing now to research the topic. (This may take 2-3 minutes.)"
        )
        st_callback_handler = demo_util.StreamlitCallbackHandler(status)
        with status:
            # STORM main gen outline
            st.session_state["runner"].run(
                topic=st.session_state["page3_topic"],
                do_research=True,
                do_generate_outline=True,
                do_generate_article=False,
                do_polish_article=False,
                callback_handler=st_callback_handler,
            )
            conversation_log_path = os.path.join(
                st.session_state["page3_current_working_dir"],
                st.session_state["page3_topic_name_truncated"],
                "conversation_log.json",
            )
            demo_util._display_persona_conversations(
                DemoFileIOHelper.read_json_file(conversation_log_path)
            )
            st.session_state["page3_write_article_state"] = "final_writing"
            status.update(label="brain**STORM**ing complete!", state="complete")


def handle_final_writing():
    if st.session_state["page3_write_article_state"] == "final_writing":
        # Initialize error counter in session state if not exists
        if "page3_error_count" not in st.session_state:
            st.session_state["page3_error_count"] = 0
            st.session_state["page3_last_error_time"] = time.time()
            
        # Reset error count if more than 30 seconds have passed
        if time.time() - st.session_state["page3_last_error_time"] > 30:
            st.session_state["page3_error_count"] = 0
            
        with st.status("Connecting information and generating final article...") as status:
            try:
                # Run STORM pipeline
                st.session_state["runner"].run(
                    topic=st.session_state["page3_topic"],
                    do_research=False,
                    do_generate_outline=False,
                    do_generate_article=True,
                    do_polish_article=True,
                    remove_duplicate=False,
                )
                st.session_state["runner"].post_run()

                # Generate combined markdown file
                article_dir = os.path.join(
                    st.session_state["page3_current_working_dir"],
                    st.session_state["page3_topic_name_truncated"]
                )
                polished_article_path = os.path.join(article_dir, "storm_gen_article_polished.txt")
                url_to_info_path = os.path.join(article_dir, "url_to_info.json")
                
                if os.path.exists(polished_article_path) and os.path.exists(url_to_info_path):
                    article_text = DemoFileIOHelper.read_txt_file(polished_article_path)
                    url_to_info = DemoFileIOHelper.read_json_file(url_to_info_path)
                    
                    # Add spaces between citations
                    article_text = re.sub(r'\](\[\d+\])', r'] \1', article_text)
                    
                    # Generate bibliography
                    bibliography = DemoTextProcessingHelper.construct_bibliography_from_url_to_info(url_to_info)
                    
                    # Combine content
                    if "# References" not in article_text:
                        combined_content = f"{article_text}\n\n# References\n\n{bibliography}"
                    else:
                        sections = article_text.split("# References")
                        combined_content = f"{sections[0].rstrip()}\n\n# References\n\n{bibliography}"
                    
                    # Save combined markdown
                    combined_md_path = os.path.join(
                        st.session_state["page3_current_working_dir"],
                        st.session_state["page3_topic_name_truncated"],
                        "combined.md",
                    )
                    DemoFileIOHelper.write_str(combined_content, combined_md_path)
                    
                    # Reset error count on success
                    st.session_state["page3_error_count"] = 0
                    
                    # Update status and state
                    status.update(label="Article generation complete!", state="complete")
                    st.session_state["page3_write_article_state"] = "prepare to show result"
                else:
                    raise FileNotFoundError("Article files not found")
                    
            except Exception as e:
                # Update error count and time
                st.session_state["page3_error_count"] += 1
                st.session_state["page3_last_error_time"] = time.time()
                
                # If too many errors, abort
                if st.session_state["page3_error_count"] >= 5:
                    status.update(label="Too many errors occurred. Aborting article generation.", state="error")
                    st.error("Article generation aborted due to multiple errors. Please try again.")
                    st.session_state["page3_write_article_state"] = "not started"
                    # Clean up any temporary files
                    if "page3_current_working_dir" in st.session_state:
                        try:
                            article_dir = os.path.join(
                                st.session_state["page3_current_working_dir"],
                                st.session_state["page3_topic_name_truncated"]
                            )
                            if os.path.exists(article_dir):
                                import shutil
                                shutil.rmtree(article_dir)
                        except Exception:
                            pass
                else:
                    error_msg = str(e)
                    if len(error_msg) > 100:  # Truncate very long error messages
                        error_msg = error_msg[:97] + "..."
                    status.update(label=f"Error during article generation: {error_msg}", state="error")


def handle_prepare_to_show_result():
    if st.session_state["page3_write_article_state"] == "prepare to show result":
        _, show_result_col, _ = st.columns([4, 3, 4])
        with show_result_col:
            if st.button("show final article"):
                st.session_state["page3_write_article_state"] = "completed"
                st.rerun()


def handle_completed():
    if st.session_state["page3_write_article_state"] == "completed":
        # display polished article
        current_working_dir_paths = DemoFileIOHelper.read_structure_to_dict(
            st.session_state["page3_current_working_dir"]
        )
        current_article_file_path_dict = current_working_dir_paths[
            st.session_state["page3_topic_name_truncated"]
        ]
        demo_util.display_article_page(
            selected_article_name=st.session_state["page3_topic_name_cleaned"],
            selected_article_file_path_dict=current_article_file_path_dict,
            show_title=True,
            show_main_article=True,
        )


def create_new_article_page():
    demo_util.clear_other_page_session_state(page_index=3)

    if "page3_write_article_state" not in st.session_state:
        st.session_state["page3_write_article_state"] = "not started"

    handle_not_started()

    handle_initiated()

    handle_pre_writing()

    handle_final_writing()

    handle_prepare_to_show_result()

    handle_completed()
