from flask import Flask, flash, redirect, render_template, request, session, abort, send_from_directory,send_file,jsonify
import pandas as pd
import numpy as np
import json
from sklearn.manifold import TSNE
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import classification_report
import os

application = Flask(__name__)

def get_tsne(df):
	df_vals = df.values
	embedded_tsne = TSNE(n_components=2).fit_transform(df_vals[:,1:-2])
	return df_vals[:,0], df_vals[:,-2], df_vals[:,-1], embedded_tsne

def get_df_for_each_label(df):
	df_vals = df.values
	final_df = np.zeros((156,5))
	for i in range(5):
		df_label = df_vals[df_vals[:,-2] == i][:,1:-2]
		pca = PCA(n_components=1, svd_solver='auto')
		df_label_pca=pca.fit_transform(df_label)
		final_df[:,i] = df_label_pca[:,0]
		new_final_df = pd.DataFrame({'Epithelium':final_df[:,0],'Stroma':final_df[:,1],'Tumor':final_df[:,2],'Necrosis':final_df[:,3],'Dysplasia':final_df[:,4]})
	return new_final_df

def MDS_func(df):
	corr_mat = 1 - df.corr().values
	embedding = MDS(n_components=2,dissimilarity='precomputed')
	embedded_data = embedding.fit_transform(corr_mat)
	return embedded_data

def mds_edges(coordinates, correlation):
	edges_list = []
	for i in range(coordinates.shape[0]):
		for j in range(i+1):
			if i != j:
				edges_list.append([coordinates[i][0],coordinates[i][1],coordinates[j][0],coordinates[j][1],np.abs(correlation[i][j])])
	return edges_list

def get_summary(df):
	df_vals = df.values
	y_true = df_vals[:,-2]
	y_pred = df_vals[:,-1]
	target_names = ["Epithelium","Stroma","Tumor","Necrosis","Dysplasia"]
	report = classification_report(y_true,y_pred,target_names=target_names,output_dict=True)
	report_df = pd.DataFrame(report).transpose()
	report_df = report_df.round({'precision':2,'recall':2,'f1-score':2,'support':2})
	labels = ['Epithelium','Stroma','Tumor','Necrosis','Dysplasia','Accuracy','Macro Avg','Micro Avg']
	report_df['labels'] = labels
	return report_df


class DataStore():
	df = None
	tsne_data = None
	idx = None
	ground = None
	predict = None
	mds_data = None
	mds_edges = None
	pcp_data = None
	summary = None
data=DataStore()

@application.route("/", methods=["GET","POST"])
def index():
	path = os.getcwd()
	df = pd.read_csv(path+'\Data\Inter.csv')
	data.df = df
	data.idx, data.ground, data.predict, data.tsne_data = get_tsne(df)
	selected_index = 0
	img_file_name = glob.glob('D:\Visualization\Project\Data\{}_*.png'.format(selected_index))
	df_for_each_label = get_df_for_each_label(df)
	data.mds_data = MDS_func(df_for_each_label)
	data.mds_edges = mds_edges(data.mds_data,df_for_each_label.corr().values)
	data.pcp_data = df_for_each_label
	data.summary = get_summary(df)
	return render_template("index.html")

@application.route("/tsne_ground", methods=["GET","POST"])
def tsne_ground():
	tsne_data = np.hstack((data.idx.reshape(-1,1),data.ground.reshape(-1,1),data.tsne_data))
	df_dict = {'data':tsne_data.tolist()}
	return jsonify(df_dict)

@application.route("/tsne_predict", methods=["GET","POST"])
def tsne_predict():
	tsne_data = np.hstack((data.idx.reshape(-1,1),data.predict.reshape(-1,1),data.tsne_data,data.ground.reshape(-1,1)))
	df_dict = {'data':tsne_data.tolist()}
	return jsonify(df_dict)

@application.route("/mds", methods=["GET","POST"])
def mds():
	mds_dict = {'data':data.mds_data.tolist(),'edges':data.mds_edges}
	return jsonify(mds_dict)

@application.route("/pcp", methods=["GET", "POST"])
def pcp():
	pcp_data = {'data':data.pcp_data.values.tolist()}
	return jsonify(pcp_data)


@application.route("/summary", methods=["GET", "POST"])
def summary():
	summary_data = {'data':data.summary.values.tolist()}
	return jsonify(summary_data)

if __name__ == "__main__":
    application.run(debug=True)

