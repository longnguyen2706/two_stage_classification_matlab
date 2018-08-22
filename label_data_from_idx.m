
function [data, label] = label_data_from_idx(total_data, total_label, idx)

	data = total_data(idx, :);
	label = total_label(idx, :);
end