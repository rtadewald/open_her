# from RealtimeTTS import TextToAudioStream, OpenAIEngine
# from langchain_community.chat_models import ChatOllama


# llm = ChatOllama(model="llama3:8b")

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())

# def dummy_generator():
#     for chunk in llm.stream("Olá, tudo bem? Quem é você?"):
#         yield chunk.content
    
# engine = OpenAIEngine(model="tts-1", voice="nova")
# stream = TextToAudioStream(engine)
# stream.feed(dummy_generator())
# # stream.feed()

# print ("Synthesizing...")
# stream.play()


if __name__ == '__main__':
    from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine, GTTSEngine

    engine = GTTSEngine() # replace with your TTS engine
    stream = TextToAudioStream(engine)
    print("here")
    # stream.feed("Hello world! How are you today?")
    stream.feed("Olá, como você está?")
    stream.play_async()

    # def dummy_generator():
    #     yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "
    #     yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."


    # # for normal use with minimal logging:

    # # test with extended logging:
    # import logging
    # # logging.basicConfig(level=logging.INFO)    
    # # engine = CoquiEngine(level=logging.INFO)


    # engine = CoquiEngine()
    # stream = TextToAudioStream(engine)
    
    # print ("Starting to play stream")
    # stream.feed(dummy_generator()).play()

    # engine.shutdown()
    