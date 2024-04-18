import logging
import os

import streamlit as st
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

logger = logging.getLogger(__name__)


@st.cache_data
def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    try:
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        ice_servers = token.ice_servers
        logger.info("ICE servers retrieved successfully.")
        return ice_servers
    except TwilioRestException as e:
        logger.error("Failed to fetch ICE servers: %s", e)
        # Handle the error gracefully, e.g., return a default server configuration
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


# ice_servers = get_ice_servers()
# print(ice_servers)
