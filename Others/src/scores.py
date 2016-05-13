import numpy as np

def calc_scores(arr1, arr2, auth_list):
	num_authors = len(auth_list);
	num_docs = len(arr1);
	confus_mat = [];
	
	for i in range(num_authors):
    	row = [];
    	for j in range(num_authors):
        	row.append(0);
    	confus_mat.append(row);
    
    for i in range(num_docs):
    	row_num = auth_list.index(arr1[i]);
    	col_num = auth_list.index(arr2[i]);
    	confus_mat[row_num][col_num] = confus_mat[row_num][col_num]+1;

    correct_preds=0;
    for i in range(num_authors):
    	correct_preds = correct_preds + confus_mat[i][i];
    accuracy = (float(correct_preds))/(float(num_docs));

    class_precs = [];
    for i in range(num_authors):
    	total_class_preds=0;
    	for j in range(num_authors):
    		total_class_preds = total_class_preds + confus_mat[j][i];
    	class_prec = (float(confus_mat[i][i]))/(float(total_class_preds));
    	class_precs.append(class_prec);

    class_recalls = [];
    for i in range(num_authors):
    	total_class_docs=0;
    	for j in range(num_authors):
    		total_class_docs = total_class_docs + confus_mat[i][j];
    	class_recall = (float(confus_mat[i][i]))/(float(total_class_docs));
    	class_recalls.append(class_recall);

    avg_macro_prec = sum(class_precs)/(float(num_authors));
    avg_macro_recall = sum(class_recalls)/(float(num_authors));
    avg_macro_fmeasure = (2.0 * avg_macro_prec * avg_macro_recall)/(avg_macro_prec + avg_macro_recall);

    print "accuracy = " + accuracy;
    print "average macro precision = " + avg_macro_prec;
    print "average macro recall = " + avg_macro_recall;
    print "average macro fmeasure = " + avg_macro_fmeasure;
    