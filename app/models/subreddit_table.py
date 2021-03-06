from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Boolean

Base = declarative_base()
class Subreddit(Base):
    __tablename__ = 'subreddits'
    MESSAGE_PREFIX = Column(String)
    STR_FIELD = Column(String)
    accept_followers = Column(Boolean)
    accounts_active = Column(Integer)
    accounts_active_is_fuzzed = Column(Boolean)
    active_user_count = Column(Integer)
    advertiser_category = Column(String)
    all_original_content = Column(Boolean)
    allow_chat_post_creation = Column(Boolean)
    allow_discovery = Column(Boolean)
    allow_galleries = Column(Boolean)
    allow_images = Column(Boolean)
    allow_polls = Column(Boolean)
    allow_prediction_contributors = Column(Boolean)
    allow_predictions = Column(Boolean)
    allow_predictions_tournament = Column(Boolean)
    allow_videogifs = Column(Boolean)
    allow_videos = Column(Boolean)
    banner_background_color = Column(String)
    banner_background_image = Column(String)
    banner_img = Column(String)
    can_assign_link_flair = Column(Boolean)
    can_assign_user_flair = Column(Boolean)
    collapse_deleted_comments = Column(Boolean)
    comment_score_hide_mins = Column(Integer)
    community_icon = Column(String)
    community_reviewed = Column(Boolean)
    created = Column(Float)
    created_utc = Column(Float)
    description = Column(String)
    description_html = Column(String)
    disable_contributor_requests = Column(Boolean)
    display_name = Column(String)
    display_name_prefixed = Column(String)
    emojis_enabled = Column(Boolean)
    free_form_reports = Column(Boolean)
    fullname = Column(String)
    has_menu_widget = Column(Boolean)
    header_img = Column(String)
    header_title = Column(String)
    hide_ads = Column(Boolean)
    icon_img = Column(String)
    id = Column(Integer, primary_key=True)
    id_subreddit = Column(String)
    is_chat_post_feature_enabled = Column(Boolean)
    is_crosspostable_subreddit = Column(Boolean)
    key_color = Column(String)
    lang = Column(String)
    link_flair_enabled = Column(Boolean)
    link_flair_position = Column(String)
    mobile_banner_image = Column(String)
    name = Column(String, unique=True)
    original_content_tag_enabled = Column(Boolean)
    over18 = Column(Boolean)
    prediction_leaderboard_entry_type = Column(String)
    primary_color = Column(String)
    public_description = Column(String)
    public_description_html = Column(String)
    public_traffic = Column(Boolean)
    quarantine = Column(Boolean)
    restrict_commenting = Column(Boolean)
    restrict_posting = Column(Boolean)
    should_archive_posts = Column(Boolean)
    show_media = Column(Boolean)
    show_media_preview = Column(Boolean)
    spoilers_enabled = Column(Boolean)
    submission_type = Column(String)
    submit_link_label = Column(String)
    submit_text = Column(String)
    submit_text_html = Column(String)
    submit_text_label = Column(String)
    subreddit_type = Column(String)
    subscribers = Column(Integer)
    title = Column(String)
    url = Column(String)
    user_can_flair_in_sr = Column(Boolean)
    user_flair_enabled_in_sr = Column(Boolean)
    user_flair_position = Column(String)
    user_flair_type = Column(String)
    user_has_favorited = Column(Boolean)
    user_is_banned = Column(Boolean)
    user_is_contributor = Column(Boolean)
    user_is_moderator = Column(Boolean)
    user_is_muted = Column(Boolean)
    user_is_subscriber = Column(Boolean)
    user_sr_flair_enabled = Column(Boolean)
    user_sr_theme_enabled = Column(Boolean)
    videostream_links_count = Column(Integer)
    whitelist_status = Column(String)
    wls = Column(Integer)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "id=%d name=%s" % (self.id, self.name)

    def __str__(self):
        return "id=%d name=%s" % (self.id, self.name)
