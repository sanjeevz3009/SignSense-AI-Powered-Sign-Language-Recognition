import logging
import os

import streamlit as st
from twilio.rest import Client

logger = logging.getLogger(__name__)

@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.
    """
    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        print("SuccessSuccessSuccessSuccess")
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    try:
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        print("SuccessSuccessSuccessSuccess2")
        return token.ice_servers
    except Exception as e:
        logger.error(f"Failed to retrieve ICE servers from Twilio: {str(e)}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]  # Fallback to Google STUN server

get_ice_servers()

import os

for key, value in os.environ.items():
    print(f'{key}: {value}')