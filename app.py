import streamlit as st
import logging
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_transformers import LongContextReorder
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# pip install streamlit langchain openai tiktoken rank_bm25 faiss-cpu

st.set_page_config(page_title='Q&A app')
st.title('Q&A app')
st.markdown('### -架空の家電量販店 お客様からの問い合わせ対応-')

# テキストデータの読み込み
raw_documents = TextLoader("./data/家電量販店Q&A.txt", encoding='utf-8').load()

### CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",  # セパレータ
    chunk_size = 300,  # チャンクの文字数
    chunk_overlap = 0,  # 重なりの最大文字数
    length_function = len, # チャンクの長さがどのように計算されるか len 文字数 
    is_separator_regex = False, 
    # セパレータが正規表現かどうかを指定 True セパレータは正規表現 False 文字列
)

## データの分割
# documentオブジェクト
documents = text_splitter.split_documents(raw_documents)

# テキストデータ　M25Retrieverに渡す前にオブジェクトから文字列に変換する必要あり。
documents_txt = [doc.page_content for doc in text_splitter.split_documents(raw_documents)]

# embeddingモデルの初期化
embedding_model = OpenAIEmbeddings() 

# llmの初期化
llm = ChatOpenAI(temperature=0) # temperature responseのランダム性0
    
def run_retriever():
    # vectorstoreの読み込み
    vectorstore = FAISS.load_local("./fiass_index", embedding_model)

    ## 初期化 retriever
    # bm25 retriever
    bm25_retriever = BM25Retriever.from_texts(documents_txt)
    bm25_retriever.k = 2

    # fiass_retriever
    fiass_retriever = vectorstore.as_retriever(
         embedding_function=OpenAIEmbeddings(), 
         search_kwargs={"k": 2}
         )
    
    # MultiQueryRetrieverの初期化
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )

    with st.expander('設定: retrieverの比率', expanded=False):
        ## retrieverの比率設定
        # bm25_retriever
        bm25_rate = st.slider('■ bm25_retrieverの比率設定', 
                                min_value=0.0, 
                                max_value=1.0,
                                value=0.3, 
                                step=0.1
                                ) 
        # fiass_retriever
        fiass_rate = st.slider('■ fiass_retrieverの比率設定', 
                                min_value=0.0, 
                                max_value=1.0,
                                value=0.5, 
                                step=0.1
                                )

        val_c = st.slider('■ 上位ランク下位ランクバランス調整', 
                            min_value=0, 
                            max_value=100,
                            value=80, 
                            step=10
                            ) 

        # multiquery_retriever
        multiquery_rate = 1- bm25_rate - fiass_rate

        st.write(f'-bm25: {bm25_rate}- -fiass: {fiass_rate}- multi_query: {multiquery_rate}')

    # ensemble retrieverの設定
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, fiass_retriever, multiquery_retriever], 
        weights=[bm25_rate, fiass_rate, multiquery_rate],
        c=val_c,
        _='''
        複数の検索結果を統合する際に、上位にランク付けされた項目と下位にランク付けされた項目の
        バランスを調整するための定数。
        「c」の値が大きいほど、上位にランク付けされた項目がより強く優先される。
        '''
    )

    # promptのtemplateの作成
    template = """
    ###Instruction###
    ・あなたは質問に対する的確な応答が得意なAIです。
    ・私たちは以下に「コンテキスト情報」を与えます。
    ・あなたのタスクはこの情報をもとに以下の「質問」に対して答えることです。
    ・良い返答には30万ドルのチップを渡します！商品の紹介であればキーワードを多く押さえた
    　商品を一つだけ選んでください。
    ・必ず日本語で答えなければなりません。
    
    ###Question### 
    {question}

    ###Context###
    {context}

    """
    # promptオブジェクトの作成　チャットモデルに情報と質問を構造化された方法で提示
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)

    
    # chat_input
    query = st.chat_input("ご用件をどうぞ")

    if query:

        # ensemble_retrieverの実行
        ensemble_docs = ensemble_retriever.get_relevant_documents(query)
        # クエリをベクトル化する際に、OpenAI Embeddingsを使用

        with st.expander('設定: chunk数の設定', expanded=False):
            # chunk数を表示
            st.write(f'■ chunk数: {len(ensemble_docs)} ■ 全chunk数: {len(documents)}')

            # ensemble_docsのchunkの数を決定
            len_chunk2 = st.slider('■ ensemble_docsのchunkの数の絞込み', 
                                min_value=0, 
                                max_value=10,
                                value=4, 
                                step=1
                                ) 

            # chunkの数を絞込み
            ensemble_docs = ensemble_docs[:len_chunk2]

            # chunk数を表示
            st.write(f'■ chunk数: {len(ensemble_docs)}')
            
        # multiquery_retrieverが生成したqueryをコマンドプロンプトにログ表示
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        
        # 精度を上げる為に関連性の低いドキュメントはリストの中央に。
        # 関連性の高いドキュメントは先頭または末尾に配置します。
        reordering = LongContextReorder()
        # ドキュメントを再配置
        reordered_docs = reordering.transform_documents(ensemble_docs)

        # LLMとPromptTemplateを連携させクエリを実行
        chain = LLMChain(llm=llm, prompt=prompt)

        # chainの実行
        response = chain({'question': query, 'context': reordered_docs})
        
        # chat表示
        with st.chat_message("user"):
            st.write(query)

        message = st.chat_message("assistant")
        message.write(response['text'])
        
        # dataの表示
        with st.expander('raw_documents', expanded=False):
            st.write(raw_documents)
        
        with st.expander('documents', expanded=False):
            st.write(documents)
        
        with st.expander('docs', expanded=False):
             st.write(ensemble_docs)
        
        with st.expander('reordered_docs', expanded=False):
            st.write(reordered_docs)
        
        with st.expander('response.context', expanded=False):
            st.write(response['context'])

def save_chroma():
    import os
    import shutil

    dir_path = './fiass_index/'

    if os.path.isdir(dir_path):
        # fiass_indexフォルダの削除
        shutil.rmtree(dir_path)
        st.write(f'{dir_path} 削除完了')

        # vectorstoreの作成
        vectorstore = FAISS.from_documents(documents, embedding_model)
        # vectorstoreの保存
        vectorstore.save_local("./fiass_index")
        st.write('vectorstoreの保存完了')

    else:
        # vectorstoreの作成
        vectorstore = FAISS.from_documents(documents, embedding_model)
        # vectorstoreの保存
        vectorstore.save_local("./fiass_index")
        st.write('vectorstoreの保存完了')

def main():
    # function名と対応する関数のマッピング
    funcs = {
        'retrieverの実行': run_retriever,
        'chroma_vectorstoreの作成': save_chroma
    }

    selected_func_name = st.sidebar.selectbox(label='項目の選択',
                                             options=list(funcs.keys()),
                                             key='func_name'
                                             )
            
    # 選択された関数を呼び出す
    render_func = funcs[selected_func_name]
    render_func()

if __name__ == '__main__':
    main()