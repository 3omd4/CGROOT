#include "mainwindow.h"
#include <QApplication>
#include <QStyleFactory>
#include <QDir>
#include <QPalette>
#include <QColor>
#include <QMessageBox>
#include <QDebug>
#include <exception>
#include <iostream>

int main(int argc, char *argv[])
{
    try {
        QApplication app(argc, argv);
        
        app.setApplicationName("Fashion-MNIST Trainer");
        app.setApplicationVersion("1.0.0");
        app.setOrganizationName("CGroot++");
        
        app.setStyle(QStyleFactory::create("Fusion"));
        
        QPalette darkPalette;
        darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::WindowText, Qt::white);
        darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
        darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
        darkPalette.setColor(QPalette::ToolTipText, Qt::white);
        darkPalette.setColor(QPalette::Text, Qt::white);
        darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::ButtonText, Qt::white);
        darkPalette.setColor(QPalette::BrightText, Qt::red);
        darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
        darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
        darkPalette.setColor(QPalette::HighlightedText, Qt::black);
        app.setPalette(darkPalette);
        
        qDebug() << "Creating MainWindow...";
        MainWindow window;
        qDebug() << "Showing MainWindow...";
        window.show();
        qDebug() << "Entering event loop...";
        
        return app.exec();
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        QMessageBox::critical(nullptr, "Fatal Error", 
            QString("The application encountered a fatal error:\n\n%1").arg(e.what()));
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        QMessageBox::critical(nullptr, "Fatal Error", 
            "The application encountered an unknown fatal error.");
        return 1;
    }
}

