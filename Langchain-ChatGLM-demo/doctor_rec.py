import os
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "./Embedding-models/text2vec-base-chinese",
}

def load_embedding_model(model_name="ernie-tiny"):
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}

    # 加载embedding model
    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def store_chroma(docs, embeddings, persist_directory='VectorStore'):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db

def chat(prompt, history=None):
    payload = {
        "prompt": prompt, "history": [] if not history else history
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(
        url='http://127.0.0.1:8000',
        json=payload,
        headers=headers
    ).json()

    return resp['response'], resp['history']

def fill_table():
    gender = input('请输入性别:')
    age = input('请输入年龄:')

    message = f'患者性别:{gender}\n' \
              f'患者年龄:{age}\n'
    return message


embeddings = load_embedding_model('text2vec3')

if not os.path.exists('VectorStore-Doctor'):
    loader = TextLoader('resources/doctor_csv_clear2.txt', encoding='utf-8')
    documents = loader.load()
    # text_spliter = CharacterTextSplitter(separator='\n')
    text_spliter = CharacterTextSplitter(separator='\n', chunk_size=1, chunk_overlap=0)
    # print(text_spliter._chunk_size)
    # print(text_spliter._chunk_overlap)
    # print(len(documents))
    # print(len(documents[0].page_content))
    split_docs = text_spliter.split_documents(documents)
    # for i in split_docs:
    #     print(len(i.page_content))
    db = store_chroma(split_docs, embeddings, 'VectorStore-Doctor')
else:
    db = Chroma(persist_directory='VectorStore-Doctor', embedding_function=embeddings)


while True:
    message = fill_table()
    query = input('User: ')
    similar_docs = db.similarity_search(query, include_metadata=False, k=3)
    system_prompt = "<系统设置>\n请记住，现在你是这家医院的前台咨询，你需要根据<用户的个人信息>以及<用户描述的症状>，结合<医院的人员信息表>，为用户推荐合适的医生和科室。\n</系统设置>\n"
    prompt = system_prompt
    # prompt += "请参考下面的例子生成你的回答：\n"
    # prompt += "<例子>\n"
#     prompt += '''**输入：**
# <用户的个人信息>如下:
# 患者性别:男
# 患者年龄:25
# <用户的症状描述>如下:
# 我今天早上突然胃疼难忍，一直没有好转，怎么办？
# 请查询下面给出的<医院人员信息表>，结合<医院的人员信息表>，为用户推荐合适的医生和科室。如果资料不足，无法回答，则回复'抱 歉，我的信息不足，无法为您推荐医生和科室。'
# <医院的人员信息表>如下：
# 表格格式为<关键词> 取值 </关键词>
# 1. <姓名>王静洁</姓名>;<科室门诊>中医儿科门诊</科室门诊>;<职称>中医师</职称>;<医生简介>中医师，硕士研究生。</医生简介>;</医生擅长>擅长中医内外治法结合治疗小儿肺脾疾病，如消化不良、功能性腹痛、咳嗽、反复上呼 吸道感染、过敏性鼻炎、腺样体肥大及中医药治疗性早熟、生长发育迟缓等疾病。</医生擅长>
# 2. <姓名>薛文英</姓名>;<科室门诊>内科门诊</科室门诊>;<职称>主任医师</职称>;<医生简介>主任医师，从事消化内科专业近20年。</医生简介>;</医生擅长>对消化内科常见病、多发病、消化不良等，急危重症及疑难病的诊治积累了丰富经验，尤其擅长胃、肠镜操作及内镜下各类疾病的操作治疗。</医生擅长>
# 3. <姓名>李小青</姓名>;<科室门诊>儿科门诊</科室门诊>;<职称>副主任医师</职称>;<医生简介>副主任医师，从事儿科临床工作13年 ，具有丰富的新生儿专科疾病诊治经验。</医生简介>;</医生擅长>在早产儿、低出生体重儿、气胸、新生儿坏死性小肠结肠炎等疾病的 救治方面有较多临床积累。</医生擅长>
# 4. <姓名>位雅莉</姓名>;<科室门诊>营养门诊/体重管理</科室门诊>;<职称>营养师</职称>;<医生简介>营养师，医学营养学研究生，注册营养师。</医生简介>;</医生擅长>妊娠期糖尿病，肥胖，高血压，高血脂等慢性病饮食营养治疗。</医生擅长>

# **输出：**
# 根据用户的个人信息和症状描述，我建议您去内科门诊找主任医师薛文英进行咨询。薛文英主任医师从事消化内科专业近20年，擅长处理消化内科常见病、多发病、消化不良等问题，对急危重症及疑难病的诊治有丰富经验。尤其擅长胃、肠镜操作及内镜下各类疾病的操作治疗。
# 您可以前往医院内科门诊寻找薛文英主任医师，向他详细描述您的症状和不适，以便得到专业的诊断和治疗建议。如果有需要，也可以先咨询前台了解具体的挂号流程和就诊时间。
# 请注意，以上建议仅基于医院人员信息表提供的资料，具体诊断还需要由医生进行详细的检查和询问。希望您早日康复，如有其他问题，请随时提问。
# '''
    # prompt += "</例子>\n"
    prompt += "<用户的个人信息>如下:\n"
    prompt += message
    prompt += "请查询下面给出的<医院人员信息表>，结合<用户的个人信息>和<用户的症状描述>，为用户推荐适合他的症状的医生和科室。如果资料不足，无法回答，则回复'抱歉，我的信息不足，无法为您推荐医生和科室。'，而不能编造信息\n" \
              "<医院的人员信息表>如下：\n"\
              "表格格式为<关键词> 取值 </关键词>\n"
    for idx, doc in enumerate(similar_docs):
        prompt += f"{idx + 1}. {doc.page_content}\n"
    prompt += "<用户的症状描述>如下:\n"
    prompt += query + '\n'
    # prompt += system_prompt 
    print(prompt)
    response, _ = chat(prompt, [])
    print("Bot: ", response)

