from dynamics import DSystem

if __name__ == "__main__":
    root = DSystem().view(mandel_center=-0.5, init_param=1.0j)
    root.mainloop()
