/*
 * Python bindings for Custom Loss Functions
 * pybind11 only; implementations live in custom_loss_functions.cpp / .h
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "custom_loss_functions.h"

namespace py = pybind11;

PYBIND11_MODULE(custom_loss_cpp, m) {
    m.doc() = "Custom loss functions for asset return prediction (C++ backend)";

    // AdjMSELoss1 (binds AdjMSELoss1Impl)
    py::class_<AdjMSELoss1Impl, std::shared_ptr<AdjMSELoss1Impl>>(m, "AdjMSELoss1")
        .def(py::init<double>(),
             py::arg("alpha") = 2.0,
             "Step-function adjustment MSE loss\n\n"
             "Parameters:\n"
             "    alpha (float): Adjustment factor for wrong direction predictions.\n"
             "                   Correct predictions are penalized by 1/alpha,\n"
             "                   wrong predictions by alpha. Default: 2.0")
        .def("forward", &AdjMSELoss1Impl::forward,
             py::arg("outputs"), py::arg("labels"),
             "Compute the loss\n\n"
             "Parameters:\n"
             "    outputs (Tensor): Model predictions\n"
             "    labels (Tensor): Ground truth labels\n"
             "Returns:\n"
             "    Tensor: Scalar loss value")
        .def("__call__", &AdjMSELoss1Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def_readwrite("alpha", &AdjMSELoss1Impl::alpha);

    // AdjMSELoss2
    py::class_<AdjMSELoss2Impl, std::shared_ptr<AdjMSELoss2Impl>>(m, "AdjMSELoss2")
        .def(py::init<double>(),
             py::arg("beta") = 2.5,
             "Sigmoid-like adjustment MSE loss\n\n"
             "Parameters:\n"
             "    beta (float): Smoothness parameter for sigmoid adjustment.\n"
             "                  Default: 2.5")
        .def("forward", &AdjMSELoss2Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def("__call__", &AdjMSELoss2Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def_readwrite("beta", &AdjMSELoss2Impl::beta);

    // AdjMSELoss3
    py::class_<AdjMSELoss3Impl, std::shared_ptr<AdjMSELoss3Impl>>(m, "AdjMSELoss3")
        .def(py::init<double>(),
             py::arg("gamma") = 0.1,
             "ReLU-like adjustment MSE loss\n\n"
             "Parameters:\n"
             "    gamma (float): Penalty scaling factor.\n"
             "                   Correct predictions penalized by gamma,\n"
             "                   wrong predictions by (1+gamma). Default: 0.1")
        .def("forward", &AdjMSELoss3Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def("__call__", &AdjMSELoss3Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def_readwrite("gamma", &AdjMSELoss3Impl::gamma);

    // AdjMAELoss1
    py::class_<AdjMAELoss1Impl, std::shared_ptr<AdjMAELoss1Impl>>(m, "AdjMAELoss1")
        .def(py::init<double>(),
             py::arg("alpha") = 2.0,
             "Step-function adjustment MAE loss\n\n"
             "Parameters:\n"
             "    alpha (float): Adjustment factor for wrong direction predictions.\n"
             "                   Default: 2.0")
        .def("forward", &AdjMAELoss1Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def("__call__", &AdjMAELoss1Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def_readwrite("alpha", &AdjMAELoss1Impl::alpha);

    // AdjMAELoss2
    py::class_<AdjMAELoss2Impl, std::shared_ptr<AdjMAELoss2Impl>>(m, "AdjMAELoss2")
        .def(py::init<double>(),
             py::arg("beta") = 2.5,
             "Sigmoid-like adjustment MAE loss\n\n"
             "Parameters:\n"
             "    beta (float): Smoothness parameter. Default: 2.5")
        .def("forward", &AdjMAELoss2Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def("__call__", &AdjMAELoss2Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def_readwrite("beta", &AdjMAELoss2Impl::beta);

    // AdjMAELoss3
    py::class_<AdjMAELoss3Impl, std::shared_ptr<AdjMAELoss3Impl>>(m, "AdjMAELoss3")
        .def(py::init<double>(),
             py::arg("gamma") = 0.1,
             "ReLU-like adjustment MAE loss\n\n"
             "Parameters:\n"
             "    gamma (float): Penalty scaling factor. Default: 0.1")
        .def("forward", &AdjMAELoss3Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def("__call__", &AdjMAELoss3Impl::forward,
             py::arg("outputs"), py::arg("labels"))
        .def_readwrite("gamma", &AdjMAELoss3Impl::gamma);
}
