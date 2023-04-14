import bempp.api

grid_1 = bempp.api.shapes.sphere(h=1)
p1_space_1 = bempp.api.function_space(grid_1,"P",1)
def testfun(x,normal,domain_index,result):
    result[0] = x[0]
grid_fun_aa = bempp.api.GridFunction(p1_space_1,fun = testfun)
c_grid = (grid_fun_aa-grid_fun_aa).coefficients
grid_fun_1 = bempp.api.GridFunction(p1_space_1,coefficients = c_grid)

grid_2 = bempp.api.shapes.sphere(h=0.1)
p1_space_2 = bempp.api.function_space(grid_1,"P",1)
import inspect
print(inspect.getsource(type(grid_fun_aa)))
#def gridfun_2(x,normal,domain_index,result):
#    result[0] = grid_fun_1.evaluate(x)
#grid_fun_2 = bempp.api.GridFunction(p1_space_2,fun = gridfun_2)
#grid_fun_2.plot()
