    def new_find_lops(self, omega_i, phi_i, n, candi_count, ii):
        f_values = np.zeros((n, candi_count[0]))
        for i in range(n):
            if i == 0:
                for t in range(n):
                    temp_value = omega_i[ii, ii] * self.candidate_graph_obj.candidate_graph.nodes[f"{0}&{t}"][
                        "observation_probability"]
                    f_values[0, t] = temp_value
            else:
                for s in range(candi_count[i]):
                    f_values[i, s] = np.max([f_values[i-1][t]+phi_i[i-1][t,s] for t in range(candi_count[i])])

        print(f_values)

        return f_values
